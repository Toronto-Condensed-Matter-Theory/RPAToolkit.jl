##### combine_chis: the per-direction bubbles -> one block matrix per q

@testset "combine_chis" begin
    nq, d   =   3, 2
    nn      =   [complex(q + 10i + 100j, q) for q in 1:nq, i in 1:d, j in 1:d]
    zz      =   -nn
    data    =   Dict{String, Any}("chi_NN" => nn, "chi_ZZ" => zz, "chi_NN_path" => nn[1:1, :, :], "chi_ZZ_path" => zz[1:1, :, :],
                                  "spin_symmetry" => [0])

    chis    =   combine_chis(data; directions = [3, 0])
    @test length(chis) == nq
    @test size(chis[1]) == (2d, 2d)
    ##### channel innermost; each block transposed and doubled (see CHI_ORIENTATION)
    for q in 1:nq, i in 1:d, j in 1:d
        @test chis[q][2(i - 1) + 1, 2(j - 1) + 1] == 2 * nn[q, j, i]
        @test chis[q][2(i - 1) + 2, 2(j - 1) + 2] == 2 * zz[q, j, i]
        @test chis[q][2(i - 1) + 1, 2(j - 1) + 2] == 0
    end
    @test length(combine_chis(data; directions = [0, 3], suffix = "_path")) == 1

    bad     =   Dict{String, Any}("chi_NN" => nn, "chi_ZZ" => zz[:, 1:1, 1:1], "spin_symmetry" => [0])
    @test_throws ErrorException combine_chis(bad; directions = [0, 3])

    ##### a file that does not record its spin symmetry: SU2 only if the model is
    legacy   =   Dict{String, Any}("chi_NN" => nn, "chi_ZZ" => zz)
    @test_throws ErrorException combine_chis(legacy; directions = [0, 3])
    @test combine_chis(legacy; directions = [0, 3], model_symmetry = "SU2") == chis
    @test_throws ErrorException combine_chis(legacy; directions = [0, 3], model_symmetry = "U1z")
    @test combine_chis(merge(legacy, Dict("bubble_spin_factorized" => [1])); directions = [0, 3]) == chis
end

##### With spin-orbit coupling every channel pair can be stored. The CHI_ORIENTATION transpose is
##### of the WHOLE matrix, so block (b, a) of the result comes from the stored (a, b) pair.
@testset "combine_chis with cross-channel pairs" begin
    nq, d       =   2, 2
    names       =   "NXYZ"
    stored      =   Dict((a, b) => randn(ComplexF64, nq, d, d) for a in 0:3 for b in 0:3)
    data        =   Dict{String, Any}("chi_" * names[a + 1] * names[b + 1] => c for ((a, b), c) in stored)
    data["spin_symmetry"] = [2]
    chis        =   combine_chis(data; directions = [0, 1, 2, 3])
    for q in 1:nq
        full    =   zeros(ComplexF64, 4d, 4d)         # as stored: row (i, a), column (j, b)
        for ((a, b), c) in stored, i in 1:d, j in 1:d
            full[4(i - 1) + a + 1, 4(j - 1) + b + 1] = c[q, i, j]
        end
        @test chis[q] == 2 .* transpose(full)
    end

    ##### U1z: the eight pairs of the {N, Z} and {X, Y} sectors are required, the rest are zero
    u1z         =   Dict{String, Any}(k => v for (k, v) in data if k in
                        ("chi_NN", "chi_NZ", "chi_ZN", "chi_ZZ", "chi_XX", "chi_XY", "chi_YX", "chi_YY"))
    u1z["spin_symmetry"] = [1]
    chis        =   combine_chis(u1z; directions = [0, 1, 2, 3])
    @test off_sector_norm(chis[1], spin_sectors("U1z", d)) == 0
    @test chis[1][4, 1] == 2 * stored[(0, 3)][1, 1, 1]
    delete!(u1z, "chi_ZN")
    @test_throws ErrorException combine_chis(u1z; directions = [0, 1, 2, 3])

    @test dress_primitives(Dict("primitives" => [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]))[2] == [0.0, 2.0, 0.0]
end
