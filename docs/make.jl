using BOCPD
using Documenter
using DocumenterVitepress

makedocs(;
    modules=[BOCPD],
    sitename="BOCPD.jl",
    authors="Carl Bittendorf and contributors",
    repo="https://github.com/CarlBittendorf/BOCPD.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Resources" => [
            "Algorithm" => "resources/algorithm.md",
            "Missing data" => "resources/missing_data.md",
            "Extending BOCPD" => "resources/extending.md",
            "API" => "resources/api.md"
        ],
    ],
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/CarlBittendorf/BOCPD.jl",
        devbranch="main",
        devurl="dev",
    ),
)

DocumenterVitepress.deploydocs(
    repo="github.com/CarlBittendorf/BOCPD.jl.git",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)