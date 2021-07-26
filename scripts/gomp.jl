function gOMP(y0::AbstractVector, Ψ::AbstractMatrix, 
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