using Documenter
using RPAToolkit

makedocs(
    build       =   "build" ,
    sitename    =   "RPAToolkit.jl"    ,
    checkdocs   =   :exports    ,
    modules     =   [RPAToolkit.Pipeline, RPAToolkit.InputParser, RPAToolkit.Interactions, RPAToolkit.Response, RPAToolkit.Preprocess, RPAToolkit.parse_model, RPAToolkit.BilinearBasis, RPAToolkit.BilinearVertex, RPAToolkit.Plotting]   ,
    pages = [
        "Introduction"              =>  "index.md",
        "Running a calculation"     =>  "running.md",
        "Conventions"               =>  "conventions.md",
        "Examples"                  =>  "examples.md",
        "API" => [
            "Pipeline"              =>  "Pipeline.md",
            "InputParser"           =>  "InputParser.md",
            "Interactions"          =>  "Interactions.md",
            "Response"              =>  "Response.md",
            "Preprocess"            =>  "Preprocess.md",
            "ParseModel"            =>  "ParseModel.md",
            "BilinearBasis"         =>  "BilinearBasis.md",
            "BilinearVertex"        =>  "BilinearVertex.md",
            "Plotting"              =>  "Plotting.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl.git",
    devbranch = "main"
)
