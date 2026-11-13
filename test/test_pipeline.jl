##### load_rpa_input, band_model and run_rpa end to end, on a synthetic bubble

@testset "pipeline" begin

    @testset "load_rpa_input resolves paths against the YAML" begin
        mktempdir() do dir
            mkpath(joinpath(dir, "configs"))
            config  =   joinpath(dir, "configs", "lattice.yml")
            write(config, "beta: 10.0\ndata_dir: ../data\ninteractions: other/ints.jld2\n")
            input   =   cd(() -> load_rpa_input(config), tempdir())       # any working directory
            data    =   joinpath(dir, "data")
            @test input["data_dir"] == data
            @test input["output"] == joinpath(data, "lattice_")
            @test input["plots"] == joinpath(data, "lattice_")
            @test input["interactions"] == joinpath(dir, "configs", "other", "ints.jld2")
            @test input["unitcell"]["julia"] == joinpath(data, "lattice_dos_scan_data.jld2")
            @test !haskey(input, "gap_equation_rpa")

            write(config, "beta: 10.0\ngap_equation_rpa:\n  dos_threshold: 0.05\n")
            input   =   load_rpa_input(config)
            @test input["data_dir"] == joinpath(dir, "configs")
            @test input["gap_equation_rpa"]["rpa_data"] == joinpath(dir, "configs", "lattice_combined_new.jld2")
            @test rpa_output_path(input) == input["gap_equation_rpa"]["rpa_data"]
        end
    end

    @testset "band_model of a UnitCell" begin
        uc, _   =   square_model()
        bm      =   band_model(uc)
        @test bm.primitives == [[1.0, 0.0], [0.0, 1.0]]
        @test Set(bm.offsets) == Set([[1, 0], [-1, 0], [0, 1], [0, -1]])
        ##### every hopping comes with its Hermitian conjugate at the opposite offset
        mats    =   Dict(bm.offsets .=> bm.matrices)
        for o in bm.offsets
            @test mats[-o] == collect(mats[o]')
        end
        @test mats[[1, 0]] ≈ -Matrix(I, 2, 2)

        ##### H(k) is TightBindingToolkit's own, on-site fields included
        honeycomb = UnitCell([[1.0, 0.0], [0.5, sqrt(3) / 2]], 2)
        AddBasisSite!(honeycomb, [0.0, 0.0])
        AddBasisSite!(honeycomb, [0.5, sqrt(3) / 6])
        S = SpinMats(1 // 2)
        t1 = Param(-1.0, 2); AddIsotropicBonds!(t1, honeycomb, 1 / sqrt(3), S[4], "t1")
        so = Param(0.2, 2);  AddIsotropicBonds!(so, honeycomb, 1.0, im .* S[3], "spin-orbit")
        CreateUnitCell!(honeycomb, [t1, so])
        @test spin_symmetry(honeycomb) == "U1z"
        ModifyFields!(honeycomb, [[0.1, 0.0, 0.3, 0.2], [0.0, 0.1, -0.3, -0.2]])
        @test spin_symmetry(honeycomb) == "none"
        bm      =   band_model(honeycomb)
        for k in ([0.0, 0.0], [1.3, -0.4], [2pi / 3, 2pi / sqrt(3)])
            @test Matrix(bloch_hamiltonian(bm, k)) ≈ FillHoppingHamiltonian(honeycomb, k) atol = 1e-13
        end
    end

    ##### chi(q) = c0 + c1 cos qx cos qy as the stored (half, per-spin) bubble; the on-site Hubbard
    ##### then goes unstable at U * 2 chi_max = 1.
    @testset "run_rpa on a synthetic bubble" begin
        mktempdir() do dir
            config  =   joinpath(dir, "square.yml")
            write(config, """
                beta: 10.0
                directions: [0, 1, 2, 3]
                mus:
                  values: [0.0]
                instability: {lower: 0.0, upper: 10.0}
                """)
            input   =   load_rpa_input(config)

            uc, hopping     =   square_model()
            save(input["unitcell"]["julia"], Dict("unit cell" => uc, "parameters" => hopping))
            params, cases   =   hubbard_coulomb_interactions(square_uc(); onsite = 1.0, range = 1.0)
            ints    =   Dict("parameters" => params, "values" => cases, "unit cell" => square_uc())
            save(input["interactions"], ints)        # no "vertex_slots": refused below

            n       =   6
            ks      =   [[2pi * i / n, 2pi * j / n, 0.0] for i in 0:n-1 for j in 0:n-1]
            path    =   [[2pi * t / 8, 0.0, 0.0] for t in 0:8]
            chi(k)  =   0.1 + 0.05 * cos(k[1]) * cos(k[2])
            stored(kset)    =   reshape(ComplexF64[chi(k) for k in kset], :, 1, 1)
            bubble  =   Dict{String, Any}("beta" => 10.0, "mu" => 0.0, "filling" => 0.5,
                                          "primitives" => [1.0 0.0; 0.0 1.0],
                                          "reciprocal" => 2pi .* [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],   # as the bare stage stores them for a 2D lattice
                                          "ks" => reduce(vcat, permutedims.(ks)), "contracted" => reduce(vcat, permutedims.(ks)),
                                          "path" => reduce(vcat, permutedims.(path)),
                                          "path_plot" => collect(range(0, 1, length = length(path))), "path_ticks" => [0.0, 1.0])
            for label in ("chi_NN", "chi_XX", "chi_YY", "chi_ZZ")
                bubble[label]           =   stored(ks)
                bubble[label * "_path"] =   stored(path)
            end
            npzwrite(input["output"] * "_beta=10.0_mu=0.0.npz", bubble)

            ##### an interaction file without the slot layout reads as the old convention: refused
            @test_throws ErrorException run_rpa(config)
            save(input["interactions"], merge(ints, Dict("vertex_slots" => VERTEX_SLOTS)))

            ##### the bubble predates `spin_symmetry`; the model is SU2, so its four diagonal pairs are all of chi
            output  =   run_rpa(config)
            entry   =   output["beta=10.0_mu=0.0"]
            @test abs(entry["Hubbard"]["critical strength"] - 1 / (2 * 0.15)) < 1e-8
            @test entry["Hubbard"]["unstable sector"] == "spin"
            @test entry["spin_symmetry"] == "SU2"
            @test entry["vertex_slots"] == VERTEX_SLOTS
            @test entry["filling"] == 0.5
            @test entry["chi_orientation"] == CHI_ORIENTATION
            @test length(entry["Hubbard_interaction"]) == n^2
            @test length(entry["Coulomb_interaction_path"]) == length(path)
            @test isfile(rpa_output_path(input))
            @test haskey(load(rpa_output_path(input)), "beta=10.0_mu=0.0")
        end
    end

    ##### Ising SOC: S_z is conserved, so chi has the {N, Z} and {X, Y} sectors, with cross terms.
    ##### Stored (half, per-spin) chi: NN = ZZ = a, NZ = ZN = c, XX = YY = b, XY = -YX = i e. With
    ##### Hubbard U = 1 the transverse sector goes first, at 2 (b + e)_max U = 1, and the
    ##### longitudinal one would need 2 sqrt(a^2 - c^2) U = 1.
    @testset "run_rpa with Ising spin-orbit coupling" begin
        mktempdir() do dir
            config  =   joinpath(dir, "ising.yml")
            write(config, """
                beta: 10.0
                directions: [0, 1, 2, 3]
                mus:
                  values: [0.0]
                instability: {lower: 0.0, upper: 10.0}
                """)
            input   =   load_rpa_input(config)

            uc, hopping     =   ising_model()
            @test spin_symmetry(uc) == "U1z"
            @test spin_symmetry(square_model()[1]) == "SU2"
            save(input["unitcell"]["julia"], Dict("unit cell" => uc, "parameters" => hopping))
            params, cases   =   hubbard_coulomb_interactions(triangular_uc(); onsite = 1.0, range = 0.0)
            save(input["interactions"], Dict("parameters" => params, "values" => cases,
                                             "unit cell" => triangular_uc(), "vertex_slots" => VERTEX_SLOTS))

            n       =   6
            ks      =   [[2pi * i / n, 2pi * j / n, 0.0] for i in 0:n-1 for j in 0:n-1]
            path    =   [[2pi * t / 8, 0.0, 0.0] for t in 0:8]
            parts   =   Dict("NN" => k -> 0.12, "ZZ" => k -> 0.12, "NZ" => k -> 0.03, "ZN" => k -> 0.03,
                             "XX" => k -> 0.1 + 0.05 * cos(k[1]), "YY" => k -> 0.1 + 0.05 * cos(k[1]),
                             "XY" => k -> 0.02im, "YX" => k -> -0.02im)
            bubble  =   Dict{String, Any}("beta" => 10.0, "mu" => 0.0, "filling" => 0.5,
                                          "primitives" => [1.0 0.0; -0.5 sqrt(3)/2],
                                          "ks" => reduce(vcat, permutedims.(ks)), "contracted" => reduce(vcat, permutedims.(ks)),
                                          "path" => reduce(vcat, permutedims.(path)),
                                          "path_plot" => collect(range(0, 1, length = length(path))), "path_ticks" => [0.0, 1.0])
            for (pair, f) in parts
                bubble["chi_" * pair]           =   reshape(ComplexF64[f(k) for k in ks], :, 1, 1)
                bubble["chi_" * pair * "_path"] =   reshape(ComplexF64[f(k) for k in path], :, 1, 1)
            end
            file    =   input["output"] * "_beta=10.0_mu=0.0.npz"

            ##### without `spin_symmetry` the file could be an old diagonal-only bubble: refused for a U1z model
            npzwrite(file, bubble)
            @test_throws ErrorException run_rpa(config)

            bubble["spin_symmetry"] = [1]
            npzwrite(file, bubble)
            entry   =   run_rpa(config)["beta=10.0_mu=0.0"]
            @test entry["spin_symmetry"] == "U1z"
            @test entry["Hubbard"]["unstable sector"] == "transverse"
            @test abs(entry["Hubbard"]["critical strength"] - 1 / (2 * (0.15 + 0.02))) < 1e-8
            chi     =   entry["combined_chis"][1]
            @test off_sector_norm(chi, spin_sectors("U1z", 1)) == 0
            @test chi[1, 4] == chi[4, 1] == 2 * 0.03
            @test chi[2, 3] == -chi[3, 2] != 0

            ##### a bubble missing one of its sector's pairs is an error, not a zero
            delete!(bubble, "chi_XY")
            npzwrite(file, bubble)
            @test_throws ErrorException run_rpa(config)
        end
    end
end
