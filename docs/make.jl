using Documenter
using RPAToolkit

makedocs(
    build       =   "build" ,
    sitename    =   "RPAToolkit.jl"    ,
    checkdocs   =   :exports    ,
    modules     =   [RPAToolkit, RPAToolkit.Pipeline, RPAToolkit.InputParser, RPAToolkit.Interactions, RPAToolkit.Response, RPAToolkit.Preprocess, RPAToolkit.SpinSectors, RPAToolkit.parse_model, RPAToolkit.BilinearBasis, RPAToolkit.BilinearVertex, RPAToolkit.Bubble, RPAToolkit.DLR, RPAToolkit.Dynamics, RPAToolkit.Plotting]   ,
    pages = [
        "Introduction"              =>  "index.md",
        "Running a calculation"     =>  "running.md",
        "Conventions"               =>  "conventions.md",
        "Examples"                  =>  "examples.md",
        "Collective modes and mean-field theory" => "collective_modes.md",
        "API" => [
            "Pipeline"              =>  "Pipeline.md",
            "InputParser"           =>  "InputParser.md",
            "Interactions"          =>  "Interactions.md",
            "Response"              =>  "Response.md",
            "Preprocess"            =>  "Preprocess.md",
            "SpinSectors"           =>  "SpinSectors.md",
            "ParseModel"            =>  "ParseModel.md",
            "BilinearBasis"         =>  "BilinearBasis.md",
            "BilinearVertex"        =>  "BilinearVertex.md",
            "Bubble"                =>  "Bubble.md",
            "DLR"                   =>  "DLR.md",
            "Dynamics"              =>  "Dynamics.md",
            "MeanFieldToolkit interface" => "MeanField.md",
            "Plotting"              =>  "Plotting.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl.git",
    devbranch = "main"
)
