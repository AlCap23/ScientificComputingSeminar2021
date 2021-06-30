module ScientificComputing

using Pluto

# Just for the sake of it
start_slides() = begin
    Pluto.run(notebook=joinpath(@__DIR__, "notebook.jl"))
end

export start_slides
end  # module ScientificComputing
