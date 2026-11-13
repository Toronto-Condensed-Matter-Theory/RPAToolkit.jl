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
    include("test_meanfield_ext.jl")
end
