using LinearAlgebra
using Plots

x = randn(100)
A = I(100) .+ 1/norm(x,2) .* randn(100, 100)
y = A*x + 1/norm(A, 2) .* randn(100)

# Solve for x 
x̂ = A \ y
# Compute the error
ϵ = x - x̂


# Make a plot and save
plot_1 = scatter(x, y, 
    ylabel = "y", label = nothing
    )

plot_2 = scatter(x, ϵ, 
    xlabel = "x", ylabel = "Error", 
    label = nothing)

plot(plot_1, plot_2, layout = (2,1))
savefig(joinpath(pwd(), "figures", "linear_solve.pdf"))

