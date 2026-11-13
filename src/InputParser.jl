module InputParser

using YAML

export load_rpa_input

"""
    load_rpa_input(filepath) -> Dict

Read an RPAToolkit YAML config and fill in its file paths.

Every path is absolute on return. Relative ones (`data_dir`, `output`, `plots`, `interactions`,
`unitcell.julia`, `gap_equation_rpa.rpa_data`) are taken relative to the
directory of the YAML file, not the working directory, and `~` is expanded. Missing ones default
to files in `data_dir` (itself defaulting to the YAML's directory), named after the YAML file
`<name>.yml`:

| key                         | default                              |
|:----------------------------|:-------------------------------------|
| `output`, `plots`           | `<data_dir>/<name>_` (a file prefix) |
| `interactions`              | `<data_dir>/<name>_interactions.jld2` (`<name>` from the `name` key if set) |
| `unitcell.julia`            | `<data_dir>/<name>_dos_scan_data.jld2` |
| `gap_equation_rpa.rpa_data` | `<data_dir>/<name>_combined_new.jld2`, only if `gap_equation_rpa` exists |
"""
function load_rpa_input(filepath::AbstractString)
    filepath = abspath(filepath)
    input = YAML.load_file(filepath)
    base_dir = dirname(filepath)

    resolve(path) = (p = expanduser(String(path)); isabspath(p) ? p : normpath(joinpath(base_dir, p)))

    file_prefix = replace(basename(filepath), r"\.ya?ml$" => "")
    base_name = get(input, "name", file_prefix)
    data_dir = resolve(get(input, "data_dir", base_dir))
    input["data_dir"] = data_dir

    default_prefix = joinpath(data_dir, file_prefix * "_")

    input["output"] = resolve(get(input, "output", default_prefix))
    input["plots"] = resolve(get(input, "plots", default_prefix))
    input["interactions"] = resolve(get(input, "interactions", joinpath(data_dir, base_name * "_interactions.jld2")))

    if !haskey(input, "unitcell")
        input["unitcell"] = Dict{Any,Any}()
    end
    unitcell = input["unitcell"]
    unitcell["julia"] = resolve(get(unitcell, "julia", joinpath(data_dir, file_prefix * "_dos_scan_data.jld2")))

    if haskey(input, "gap_equation_rpa")
        gap = input["gap_equation_rpa"]
        gap["rpa_data"] = resolve(get(gap, "rpa_data", joinpath(data_dir, file_prefix * "_combined_new.jld2")))
    end

    return input
end

end
