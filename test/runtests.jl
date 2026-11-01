using RPAToolkit, TightBindingToolkit
using LinearAlgebra, DelimitedFiles, Random, JLD2, NPZ
using Test

include("models.jl")
include("fixtures.jl")

@testset "RPAToolkit.jl" begin
    include("test_bubble_symmetry.jl")
    include("test_bilinear_vertex.jl")
    include("test_bilinear_basis.jl")
    include("test_interactions.jl")
    include("test_response.jl")
    include("test_preprocess.jl")
    include("test_pipeline.jl")
end
