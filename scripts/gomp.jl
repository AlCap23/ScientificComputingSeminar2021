using LinearAlgebra: similar
using LinearAlgebra, SparseArrays, Random, Statistics
using Plots, Plots.PlotMeasures

function gomp(y0::AbstractVector, Ψ::AbstractMatrix, 
    K::Int = 2, S::Int = 1; 
    max_iter::Int = 100, ϵ::Real = eps())

	# Get the dimensions
	m = length(y0)
	m_psi, n = size(Ψ)
	# Assert the dimensionality
	@assert m == m_psi "Please provide consistent input sizes"
	# Assert the selector
	@assert S <=  min(K, m/K) "S <= min(K, m/K)"
	
	# Normalize
	ψ = deepcopy(Ψ)
    # Inplace
	normalize!.(eachcol(ψ), 2)
	
	
	# Preconditioning
	P = ψ'pinv(ψ*ψ')
	# New matrix
	ψ = P*ψ
	y = P*y0
	# Iteration
	iters = 0
	
	# Support
	Λ = zeros(Bool, n)
	u = zeros(eltype(y), n)
	r = y
	amps = zeros(eltype(y), n)
	
	# Find the magnitudes
	for i in 1:max_iter
		# Compute the similarities via magnitude
		amps .= ψ'r
		# Get the largest entry
		idx = sortperm(abs.(amps), rev = true)[1:S]
		# Update the support 
		Λ[idx] .= true
		# Update the coefficients
		u[Λ] .= (y' / ψ')[1, Λ]
		# Update r
		r .= y - ψ*u
		# Convergence
		(norm(r,2) < ϵ || sum(Λ) >= K) && break
	end
	
    # Last time to get the right coefficients
	u[Λ] .= (y0' / Ψ')[1, Λ]
    
	return u
end

## Test data
Random.seed!(1111) # Lucky
x̂ = sprandn(Float64, 100, 0.1)
k_opt = Int(norm(x̂, 0))
A = randn(Float64, 300, 100)
y = A*x̂

# Evaluate
x = gomp(y, A, k_opt)

# Check for different noise levels
σ = var(A, dims = 2)
Ã = similar(A)
# Estimate and return sparsity
sp_n = map(0:0.5:10.0) do i
	Ã = A
	for j in 1:size(A, 2)
		Ã[:,j] .+= σ[j]*i*randn(length(y))
	end
	ỹ = Ã*x̂
	norm(gomp(ỹ, Ã, k_opt), 0)
end


pl_1 = scatter(x̂, label = "Ground Truth", markeralpha = 0.5, xlabel = "Index", ylabel = "x", legend = :bottomright)
scatter!(x, label = "Estimated", markershape = :cross, markersize = 12, markerstrokewidth = 5)
savefig(joinpath(pwd(), "figures", "gomp.pdf"))

pl_2 = scatter((0:0.01:1.0) .* 100.0, sp_n, xlabel = "Noise Level [% of Variance]", ylabel = "Nonzero Entries", label = "Recovered", legend = :bottomleft)
plot!([0; 100], [k_opt; k_opt], linestyle = :dash, label = "Ground Truth")
savefig(joinpath(pwd(), "figures", "gomp_noise.pdf"))

plot(pl_1, pl_2, layout = (1,2), size = (900, 300), margins = 5mm)
savefig(joinpath(pwd(), "figures", "merged.pdf"))
