using MultiOptim
using Documenter

makedocs(;
    modules=[MultiOptim],
    authors="Robert Feldt",
    repo="https://github.com/robertfeldt/MultiOptim.jl/blob/{commit}{path}#L{line}",
    sitename="MultiOptim.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://robertfeldt.github.io/MultiOptim.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/robertfeldt/MultiOptim.jl",
)
