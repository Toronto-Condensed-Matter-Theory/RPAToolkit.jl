using RPAToolkit, TightBindingToolkit
using LinearAlgebra, DelimitedFiles, Random, JLD2, NPZ, YAML
using Test

include("models.jl")
include("fixtures.jl")

@testset "RPAToolkit.jl" begin
    include("test_bubble_symmetry.jl")
    include("test_spin_bubble.jl")
    include("test_bubble.jl")
    include("test_dynamics.jl")
    include("test_rpa_realspace.jl")
    include("test_bilinear_vertex.jl")
    include("test_bilinear_basis.jl")
    include("test_interactions.jl")
    include("test_response.jl")
    include("test_preprocess.jl")
    include("test_pipeline.jl")
    ##### The MeanFieldToolkit extension needs MeanFieldToolkit, which the package's own test environment
    ##### does not carry: run it in the example environment,
    #####     julia --project=examples/MeanField test/test_meanfield_ext.jl
    if Base.find_package("MeanFieldToolkit") !== nothing
        include("test_meanfield_ext.jl")
    else
        @info "MeanFieldToolkit is not in this environment: skipping test_meanfield_ext.jl."
    end
end
