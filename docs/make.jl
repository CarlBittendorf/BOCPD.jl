using BOCPD
using Documenter
using DocumenterVitepress

makedocs(;
    modules = [BOCPD],
    sitename = "BOCPD.jl",
    authors = "Carl Bittendorf and contributors",
    repo = "https://github.com/CarlBittendorf/BOCPD.jl",
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Algorithm" => "algorithm.md",
        "Missing data" => "missing_data.md",
        "Extending BOCPD" => "extending.md",
        "API reference" => "api.md",
    ],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/CarlBittendorf/BOCPD.jl",
        devbranch = "main",
        devurl = "dev",
    ),
)

DocumenterVitepress.deploydocs(
    repo = "github.com/CarlBittendorf/BOCPD.jl.git",
    devbranch = "main",
    push_preview = true,
)