##### load_rpa_input, parse_unitcell and run_rpa end to end, on a synthetic bubble (no TRIQS)

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
            @test input["unitcell"]["triqs"] == joinpath(data, "lattice_triqs.npz")
            @test !haskey(input, "gap_equation_rpa")

            write(config, "beta: 10.0\ngap_equation_rpa:\n  dos_threshold: 0.05\n")
            input   =   load_rpa_input(config)
            @test input["data_dir"] == joinpath(dir, "configs")
            @test input["gap_equation_rpa"]["rpa_data"] == joinpath(dir, "configs", "lattice_combined_new.jld2")
            @test rpa_output_path(input) == input["gap_equation_rpa"]["rpa_data"]
        end
    end

    @testset "parse_unitcell" begin
        uc, _   =   square_model()
        out     =   parse_unitcell(uc)
        @test out["units"] == [1.0 0.0; 0.0 1.0; 0.0 0.0]
        offsets =   [Tuple(c) for c in eachcol(out["hopping offsets"])]
        @test Set(offsets) == Set([(1, 0), (-1, 0), (0, 1), (0, -1)])
        ##### every hopping comes with its Hermitian conjugate at the opposite offset
        mats    =   Dict(o => out["hopping matrices"][i, :, :] for (i, o) in enumerate(offsets))
        for o in offsets
            @test mats[.-o] == collect(mats[o]')
        end
        @test mats[(1, 0)] ≈ -Matrix(I, 2, 2)
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
            save(input["interactions"], Dict("parameters" => params, "values" => cases, "unit cell" => square_uc()))

            n       =   6
            ks      =   [[2pi * i / n, 2pi * j / n, 0.0] for i in 0:n-1 for j in 0:n-1]
            path    =   [[2pi * t / 8, 0.0, 0.0] for t in 0:8]
            chi(k)  =   0.1 + 0.05 * cos(k[1]) * cos(k[2])
            stored(kset)    =   reshape(ComplexF64[chi(k) for k in kset], :, 1, 1)
            bubble  =   Dict{String, Any}("beta" => 10.0, "mu" => 0.0, "filling" => 0.5,
                                          "primitives" => [1.0 0.0; 0.0 1.0],
                                          "reciprocal" => 2pi .* [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],   # as TRIQS stores them for a 2D lattice
                                          "ks" => reduce(vcat, permutedims.(ks)), "contracted" => reduce(vcat, permutedims.(ks)),
                                          "path" => reduce(vcat, permutedims.(path)),
                                          "path_plot" => collect(range(0, 1, length = length(path))), "path_ticks" => [0.0, 1.0])
            for label in ("chi_NN", "chi_XX", "chi_YY", "chi_ZZ")
                bubble[label]           =   stored(ks)
                bubble[label * "_path"] =   stored(path)
            end
            npzwrite(input["output"] * "_beta=10.0_mu=0.0.npz", bubble)

            output  =   run_rpa(config)
            entry   =   output["beta=10.0_mu=0.0"]
            @test abs(entry["Hubbard"]["critical strength"] - 1 / (2 * 0.15)) < 1e-8
            @test entry["filling"] == 0.5
            @test entry["chi_orientation"] == CHI_ORIENTATION
            @test length(entry["Hubbard_interaction"]) == n^2
            @test length(entry["Coulomb_interaction_path"]) == length(path)
            @test isfile(rpa_output_path(input))
            @test haskey(load(rpa_output_path(input)), "beta=10.0_mu=0.0")
        end
    end
end
