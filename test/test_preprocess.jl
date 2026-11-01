##### combine_chis: the per-direction bubbles -> one block matrix per q

@testset "combine_chis" begin
    nq, d   =   3, 2
    nn      =   [complex(q + 10i + 100j, q) for q in 1:nq, i in 1:d, j in 1:d]
    zz      =   -nn
    data    =   Dict{String, Any}("chi_NN" => nn, "chi_ZZ" => zz, "chi_NN_path" => nn[1:1, :, :], "chi_ZZ_path" => zz[1:1, :, :])

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

    bad     =   Dict{String, Any}("chi_NN" => nn, "chi_ZZ" => zz[:, 1:1, 1:1])
    @test_throws ErrorException combine_chis(bad; directions = [0, 3])

    @test dress_primitives(Dict("primitives" => [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]))[2] == [0.0, 2.0, 0.0]
end
