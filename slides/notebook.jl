### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ a78a6056-49dc-4315-b12a-f290a658fc2b
begin
	using Pluto
	using PlutoUI
	
	using LinearAlgebra
	using Statistics
	using Plots
	plotly()
	
	# Some useable definitions for the presentation
	# Taken from https://andreaskroepelin.de/blog/plutoslides/
	struct Foldable{C}
    	title::String
    	content::C
	end

	function Base.show(io, mime::MIME"text/html", fld::Foldable)
    	write(io,"<details><summary>$(fld.title)</summary><p>")
    	show(io, mime, fld.content)
    	write(io,"</p></details>")
	end
	
	struct TwoColumn{L, R}
    	left::L
    	right::R
	end

	function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
    	write(io, """<div style="display: flex;"><div style="flex: 50%;">""")
    	show(io, mime, tc.left)
    	write(io, """</div><div style="flex: 50%;">""")
    	show(io, mime, tc.right)
    	write(io, """</div></div>""")
	end
	nothing
end

# ╔═╡ 11152d22-899f-4a01-bec9-8ffe887306ad
html"<button onclick=present()>Present</button>"

# ╔═╡ 7a5e88a0-cdc0-11eb-23d1-8b1a18ca5072
md""" # Julia

Outline
+ Julia as a Programming Language
+ Multiple Dispatch vs. Object Oriented Programming
+ Matching Pursuit and Speed-Ups
"""

# ╔═╡ afdf3c0f-8f09-425a-9ac9-94d5944c6e91
md""" # Preface
![](https://imgs.xkcd.com/comics/real_programmers.png)

Everyone is entitled to their personal opinion, workflow and preferences. The focus should be on the scientific progress.
"""

# ╔═╡ b0425c95-3eee-4461-9a48-3727fc2f6832
md""" # Julia : Raison d'être and Goals [[5]](https://julialang.org/blog/2012/02/why-we-created-julia/)

*In short, because we are greedy.*

+ The speed of C
+ The dynamism of Ruby
+ The obvious syntax of Matlab
+ The generalizability of Python
+ The adhesivity of the Shell
+ ...


"""

# ╔═╡ 1488e403-7d51-4bc5-b914-19b98df9ed51
md""" # Julia - A fresh approach to numerical computation [[1]](https://arxiv.org/abs/1411.1607)

Draft in high level language $\mapsto$ Reimplement in low level Language

Additional effort:
+ Map datastructures correctly
+ Ensure composability of datastructures and functions or methods
+ Implement interfaces to Open Source Projects
+ Add functionality to these
+ ...

This is known as the **Two Language Problem** in computer science.
"""

# ╔═╡ b9fd4165-9e51-48b1-81f0-4c5812ad26b9
begin
	text_c = md""" _C_
	
	+ Static Typed
	+ Compiled
	+ Highly performant
	
	""";
	text_cpp = md""" _C++_
	
	+ Object Oriented
	+ Compiled
	+ Highly performant
	
	""";
	
	text_python = md""" _Python_
	
	+ Object Oriented
	+ Dynamicly Interpreted
	+ Performant (?)
	
	"""
	
	text_julia = md""" _Julia_
	+ Dynamicly Typed
	+ Just-In-Time Compiled
	+ Highly performant
	"""
	
	# Create a dict
	comp_lang = Dict(
		"text_c" => text_c, 
		"text_cpp" => text_cpp,
		"text_python" => text_python
		)
	nothing
end

# ╔═╡ d50f89d9-06fe-402f-8613-d71591bdf9d2
md""" Let's compare Julia with $(@bind comparision_language_1 html"<select><option value='text_c'>C</option><option 		value='text_cpp'>C++</option><option 	value='text_python'>Python</option</select>") 
"""

# ╔═╡ 19815c4f-b3ec-4708-aab8-3219b4ac5a77
TwoColumn(comp_lang[comparision_language_1], text_julia)

# ╔═╡ b7d2a5f4-c6f9-43ef-9beb-38f833b5d13c
md""" ### The need for speed
"""

# ╔═╡ 2790a41c-cb5b-4eed-bb6e-7687b7c01824
# Consider the function 
f(x,y) = exp.(-(x-y).^2)

# ╔═╡ 4ccb9374-f661-47b2-8f25-3818c78d5cf3
Foldable(
	"Caveat",
md"""
Most commonly, new users consider Julia to be slow due to a first function call. This is due to the compilation of each method happening at compile level. 

```
f(x) = sum(x.^2)
x = randn(Float64, 10000);

# First time call, the compiler sorts of the method to use and compiles it
@time f(x)
  0.070206 seconds (291.00 k allocations: 17.744 MiB, 99.93% compilation time)
10105.128612847464 # <- Result

# Second time call
@time f(x)
  0.000035 seconds (3 allocations: 78.219 KiB) # <- See the drastic reduction here
10105.128612847464 # <- Result

```
"""
	)

# ╔═╡ 014a0e4c-abaf-445d-9de8-18cc1fcf7e1d
begin
	# For two arrays
	x = fill(1, 100000)
	x̂ = randn(100000)
end

# ╔═╡ fd8dd3db-aeb6-4105-b4c2-c9eca716ebed
Foldable("Referecing", 
	md"""
	By default, we reference rather than copy. Consider
	
	```
	x = randn(10);
	x[1] = 1; # Set the first element to 1
	v = x[1] # Set v <- x[1]
	x[1] = 2 # Set x[1] to 2
	v # Will return 2
	```
	"""
	)

# ╔═╡ e158522d-fe91-49fb-bf98-1fcf909f0223
md""" The error above is due to the missing vectorization of the function `f`.
We can add this quite easily.
"""

# ╔═╡ 363eebd3-0c8d-49c3-9d8a-9e6aaf6fce8b
Foldable("Memory Layout", 
	md"""
	Julia is **column major** oriented. Meaning internally, the entries of an array get stored
	
	```
	x[1,1] x[2,1] x[3,1] x[1,2] x[2,2] x[3,2] ...
	```
	
	for speed ups always try to keep this in mind.
	"""
	)

# ╔═╡ bd0fe52b-444e-442c-89c5-ba84d8bad709
begin
	
	md""" ### Benchmarks
	![Julia Microbenchmarks](https://julialang.org/assets/benchmarks/benchmarks.svg)
	Microbenchmarks of Julia vs. different Languages as currently available [here](https://julialang.org/benchmarks/)
	
	**BUT** This is an old plot. Be careful.
	
	"""
	
	
end

# ╔═╡ bdb0db4b-ac6d-4311-b506-55e82b7974e0
Foldable("Just In Time (JIT) Compilation", 
	md"""
	Julia is just in time compiled, using [LLVM](https://llvm.org/) under the hood. Similar approaches exist for [Matlab](https://de.mathworks.com/products/matlab/matlab-execution-engine.html) and [Python](https://doc.pypy.org/en/latest/architecture.html#jit-compiler). However, these are optional oppose to Julia, where it is the standard. 
	"""
	)

# ╔═╡ f51666f4-2896-4ba6-8f92-63795fb43d3a
md""" # Batteries included

Within the [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) package - included in the standard library - we have BLAS and LAPACK at our fingertips.
"""

# ╔═╡ c78dee8d-9ff2-441f-9f5b-fc60f413869a
# Simple dot
BLAS.dot(10, fill(1.0, 10), 1, fill(1.0, 20), 2)

# ╔═╡ 3d110ae2-c8d0-4355-bccc-7d7a89224453
# a*x + y
BLAS.axpy!(2, [1;2;3], [4;5;6])

# ╔═╡ 606e95e0-f8d2-47dd-964c-fc33b72a91d1
# Solve A X = B via LU(A) and overwrites B with the solution
begin 
	A = randn(3,3)
	B = A * [1 0 0; 0 0 1; 0 1 0] 
	LAPACK.gesv!(A, B)
end

# ╔═╡ ff06c5fe-500b-496e-8801-d399f1b2f394
md""" # Multiple Dispatch
"""

# ╔═╡ 62d14388-0616-4ab3-af97-82e31d380157
TwoColumn(
	md""" **Functional**
	
	_Global_ Namespace
	
	$f \in \mathcal{F}$
	
	_Unique_ Functions
	
	$f(x_1, x_2, x_3, \dots)$
	
	Results in
	
	```
	z = 3+i4
	r = 0.1
	complex_real_add(z, r)
	```
	""",
	
	md""" **Object Oriented**
	
	_Local_ Namespace
	
	$f \in \mathcal{O}$
	
	_Unique_ Functions in Namespace
	
	$o.f(x_1, x_2, x_3, \dots)$
	
	Can result in
	```
	z = 3+i4
	r = 0.1
	z + 0.1
	```
	"""
)

# ╔═╡ 148805bf-e3c1-497b-a776-fd997a6d7acd
md""" **Multiple Dispatch**

Global namespace

$f \in \mathcal{F}$

With unique arguments 

$f : \mathcal{X}_1 \times \mathcal{X}_2 \dots \mapsto \mathcal{Y}$
"""

# ╔═╡ 1d53fe22-000f-4ee4-9c65-c4f755dd3ce6
begin 
	
	_xx = md"Consider the expressive power [[2]](https://www.youtube.com/watch?v=kc9HwsxE1OY)
	
| | Functional | OOP | Multiple Dispatch
:--- | --- | ----- | ---:
Dispatch arg. | 0 | 1   |   N 
Expressive Power | const. | linear | exponential
"
	Foldable("Is this useful?", _xx)
end

# ╔═╡ d1f967e2-bea9-45f4-8760-eb22d1908bf8

	Foldable("But why exactly is this useful?", md" **EFFECTIVE REUSE AND EXTENABILITY OF CODE** [[3]](https://www.oxinabox.net/2020/02/09/whycompositionaljulia.html)
	
	```
using StatsBase  # exports predict
using Foo  # overloads `StatsBase.predict(::FooModel)
using Bar  # overloads `StatsBase.predict(::BarModel)
training_data, test_data = ...
mbar = BarModel(training_data)
mfoo = FooModel(training_data)
evaluate(predict(mbar), test_data)
evaluate(predict(mfoo), test_data)
```
	")

# ╔═╡ 66079375-24c7-4e27-ba41-08fd5634fcd7
md""" 
**Example** : Dual Numbers

$z = a + b \epsilon$  

with

$\epsilon^2 = 0$
"""

# ╔═╡ 7b751a69-72aa-4543-b89a-0075a48f3bf0
# Type definition
struct DualNumber{T} <: Number
	a::T
	b::T
end

# ╔═╡ 1d55314e-7cb4-4be7-8e44-94c4a95739dd
# Addition
Base.:+(x::DualNumber{T}, y::DualNumber{T}) where T = DualNumber(x.a + y.a, x.b + y.b) 

# ╔═╡ 55dfc8e4-efc5-4eb4-8534-c482d43059ae
z = DualNumber(1.0, 0.2)

# ╔═╡ 481e1757-cbd9-4b52-b6d7-b828934e0a74
z + z

# ╔═╡ 1fb93892-2ab8-4e49-9076-1517afb2299d
# Multiplication
Base.:*(x::DualNumber, y::DualNumber) = DualNumber(
	x.a*y.a, x.a*y.b+y.a*x.b
	)

# ╔═╡ 93bca010-ef25-4672-a601-b42fbae73bd3
z*z # Works

# ╔═╡ 987772c6-35e6-4340-b8ad-65eb6c5521dc
(z^3 + z^2)

# ╔═╡ ebc190c1-aa57-4ab0-8f6f-958eaae4008b
begin
	# Add multiplication
	Base.:*(x::Number, y::DualNumber{T}) where T = DualNumber(convert(T, x)*y.a, convert(T,x)*y.b)
	Base.:*(y::DualNumber{T}, x::Number) where T = DualNumber(convert(T, x)*y.a, convert(T,x)*y.b)
	# Add subtraction
	Base.:-(x::Number, y::DualNumber{T}) where T = DualNumber(convert(T,x)-y.a, -y.b)
	Base.:-(y::DualNumber{T}, x::Number) where T = DualNumber(y.a-convert(T,x), y.b)
	Base.:-(x::DualNumber, y::DualNumber) = x+(-1*y)
end

# ╔═╡ 892e2207-b7e0-410c-a22e-93aa185ea858
f(x) = x^2 - 3*x

# ╔═╡ 7584eb8c-9e6a-43cc-859a-f2940adadad0
# Lets see what happens
f(1f0, 2)

# ╔═╡ d4ec78eb-10ee-4b0f-812f-0c79f5dfce91
# And under the hood
@code_lowered f(0.2, 3.0)

# ╔═╡ 70764379-8132-4ca3-aee6-87871c2132a2
# Can we get it down? Hide at first
function fastf(x::AbstractVector{X}, y::AbstractVector{Y}) where {X, Y}
	# We want a common type
	T = promote_type(X, Y)
	# Convert the input to a common type
	x = convert.(T, x)
	y = convert.(T, y)
	# Preallocation of the result
	res = Vector{T}(undef, length(x))
	# Vectorize
	# We assume length(x) == length(y) -> Errors are not catched!
	# Additionally we add the @simd macro to instruct the compiler
	# to parallelize
	@inbounds for i in 1:length(x)
		# If we want, we could also use the @fastmath macro here
		# which does not improve our performance ( in this case )
		res[i] = f(x[i], y[i])
	end
	# The return argument
	return res
end

# ╔═╡ 76a0d6b9-f6e6-4e68-9607-17be0d8b8c19
@time fastf(x, x̂)

# ╔═╡ 0c909529-b296-4a61-a369-e0d129b37501
# Mutating function and even faster because we assume the same type
function fastf!(res::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
	# Vectorize
	# We assume length(x) == length(y) -> Errors are not catched!
	# Additionally we add the @simd macro to instruct the compiler
	# to parallelize
	@inbounds @simd for i in 1:length(x)
		# If we want, we could also use the @fastmath macro here
		# which does not improve our performance ( in this case )
		res[i] = f(x[i], y[i])
	end
	# The return argument
	return 
end

# ╔═╡ e70fd29d-6ee0-4011-9f45-df4d4e72d6d4
begin
	# Benchmarking
	mybench(f::Function, args...) = @time f(args...)
	#@time f(x, x̂)
	mybench(f, x, x̂)
	mybench(fastf, x, x̂) 
	res = similar(x̂)
	mybench(fastf!, res, convert.(eltype(x̂), x), x̂)
end

# ╔═╡ 97306279-df1f-4bb2-9d51-8e6345808100
f(z)

# ╔═╡ fb3490f1-0b3a-44b0-9bea-50cb5e530c15
md""" ### Matchig Pursuit [[4]](https://en.wikipedia.org/wiki/Matching_pursuit)

$\min_\Xi \quad \lVert \Xi \rVert_0, \quad \text{s.t.} \quad Y = \Psi(X) \Xi$

![Example](https://upload.wikimedia.org/wikipedia/commons/3/38/Orthogonal_Matching_Pursuit.gif)


**Pseudo-Code**

+ Compare each element of the normalized dictionary $\Psi(X)$ to the signal $Y$ via the inner product
+ Use the largest resemblance as a coefficient $\xi = \Xi_{i,j}$
+ Subtract the recovered signal and repeat until converged


We will use a more sophisticated version [based on this paper](https://asp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13634-020-00680-9.pdf) where our goal is to find the right support $\Lambda$ of the coefficient vector.


*Note*

`LinearAlgebra` is already imported
"""

# ╔═╡ 34d14afe-2194-43f0-a829-15276ecca883
# Define the algorithm
function gOMP(y0::AbstractVector, Ψ::AbstractMatrix, K::Int = 2, S::Int = K; max_iter::Int = 100, ϵ::Real = eps())
	# Get the dimensions
	m = length(y0)
	m_psi, n = size(Ψ)
	# Assert the dimensionality
	@assert m == m_psi "Please provide consistent input sizes"
	# Assert the selector
	@assert S <=  min(K, m/K) "S <= min(K, m/K)"
	
	# Normalize
	ψ = deepcopy(Ψ)
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
		if norm(r,2) < ϵ || sum(Λ) >= S
			# Just for debug
			#@debug "Early break after $i iterations with $(norm(r,2))"
			break
		end
	end
	# Last time to get the right coefficients
	u[Λ] .= (y0' / Ψ')[1, Λ]
	return u
	
end

# ╔═╡ 2ebc2b89-27f2-4abc-816e-7950bce2267f
# Generate some test data
begin
	# Independent variable
	t = 0.0:0.1:10.0; 
	# Signal
	y = 3.0*sin.(t).*exp.(-t./50.0) + 4.0*cos.(t) - t;
	# Dictionary
	ψ = [
		t t.^2 t.^3 sin.(t) ones(eltype(y), length(y)) cos.(t) exp.(t) sin.(t).*exp.(-t./50.0)
	];
	nothing
end

# ╔═╡ f7451fdb-b5ac-4bf7-82f2-fed46ed98e99
md""" Lets explore the data by adding a slider for the sparsity and a noise level.

Sparsity : $(@bind k Slider(1:size(ψ, 2)))

Noise : $(@bind n Slider(0.0:0.005:0.1))

"""

# ╔═╡ dac7e4e0-2532-45ff-8b0f-1d8a92e90ccf


# ╔═╡ 1679b86e-13ef-48c4-9a05-613387c213f9
# What about parallism?

# Simple 

function gOMP(Y::AbstractMatrix, Ψ::AbstractMatrix, args...; kwargs...)
	
	# we know that the coefficients can be derived independent
	A = zeros(size(Ψ, 2), size(Y, 1)) # Init
	#Threads.@threads for i in 1:size(Y, 1)
	for i in 1:size(Y, 1)
		A[:, i] .= gOMP(Y[i, :], Ψ, args...; kwargs...)
	end
	return A
end

# ╔═╡ fb971894-a833-4955-93b0-da68c4a71db0
begin 
	ynoise = y .+ n*mean(y, dims = 2).*randn(size(y))
	a = gOMP(ynoise, ψ, k)
	plot(t, ynoise, label = "Original Signal with $n percent noise", title = "Fit")
	plot!(t, ψ*a, label = "Recovered with  $(sum(abs.(a) .> 1e-10)) / $k elements ")
	annotate!([(0.0, -14.0, Plots.text("Coefficients : $(round.(a, digits = 2))", 10, :black, :left))])

	
end

# ╔═╡ 6e8805fe-05e8-49b5-a231-1efb995da173
Y = vcat([(y+5e-1*randn(size(y)))' for i in 1:100]...)

# ╔═╡ 05cccc05-89a8-48c9-b1ac-a8ff6b5089a9
@time gOMP(Y, ψ, 4)

# ╔═╡ 1d3939e1-bcec-46fd-84e0-b84abfbb3dc9
md""" ### Ecosystem & Package Development

[JuliaHub](https://juliahub.com/ui/Packages) provides us with a nice, searchable database for all registered packages.

As an example for Package Development, we can have a look at [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl).
"""

# ╔═╡ Cell order:
# ╟─a78a6056-49dc-4315-b12a-f290a658fc2b
# ╟─11152d22-899f-4a01-bec9-8ffe887306ad
# ╟─7a5e88a0-cdc0-11eb-23d1-8b1a18ca5072
# ╟─afdf3c0f-8f09-425a-9ac9-94d5944c6e91
# ╟─b0425c95-3eee-4461-9a48-3727fc2f6832
# ╟─1488e403-7d51-4bc5-b914-19b98df9ed51
# ╟─b9fd4165-9e51-48b1-81f0-4c5812ad26b9
# ╟─d50f89d9-06fe-402f-8613-d71591bdf9d2
# ╟─19815c4f-b3ec-4708-aab8-3219b4ac5a77
# ╟─b7d2a5f4-c6f9-43ef-9beb-38f833b5d13c
# ╠═2790a41c-cb5b-4eed-bb6e-7687b7c01824
# ╠═7584eb8c-9e6a-43cc-859a-f2940adadad0
# ╟─4ccb9374-f661-47b2-8f25-3818c78d5cf3
# ╠═d4ec78eb-10ee-4b0f-812f-0c79f5dfce91
# ╠═014a0e4c-abaf-445d-9de8-18cc1fcf7e1d
# ╟─fd8dd3db-aeb6-4105-b4c2-c9eca716ebed
# ╠═76a0d6b9-f6e6-4e68-9607-17be0d8b8c19
# ╟─e158522d-fe91-49fb-bf98-1fcf909f0223
# ╠═70764379-8132-4ca3-aee6-87871c2132a2
# ╠═0c909529-b296-4a61-a369-e0d129b37501
# ╠═e70fd29d-6ee0-4011-9f45-df4d4e72d6d4
# ╟─363eebd3-0c8d-49c3-9d8a-9e6aaf6fce8b
# ╟─bd0fe52b-444e-442c-89c5-ba84d8bad709
# ╟─bdb0db4b-ac6d-4311-b506-55e82b7974e0
# ╟─f51666f4-2896-4ba6-8f92-63795fb43d3a
# ╠═c78dee8d-9ff2-441f-9f5b-fc60f413869a
# ╠═3d110ae2-c8d0-4355-bccc-7d7a89224453
# ╠═606e95e0-f8d2-47dd-964c-fc33b72a91d1
# ╟─ff06c5fe-500b-496e-8801-d399f1b2f394
# ╟─62d14388-0616-4ab3-af97-82e31d380157
# ╟─148805bf-e3c1-497b-a776-fd997a6d7acd
# ╟─1d53fe22-000f-4ee4-9c65-c4f755dd3ce6
# ╟─d1f967e2-bea9-45f4-8760-eb22d1908bf8
# ╟─66079375-24c7-4e27-ba41-08fd5634fcd7
# ╠═7b751a69-72aa-4543-b89a-0075a48f3bf0
# ╠═1d55314e-7cb4-4be7-8e44-94c4a95739dd
# ╠═55dfc8e4-efc5-4eb4-8534-c482d43059ae
# ╠═481e1757-cbd9-4b52-b6d7-b828934e0a74
# ╠═1fb93892-2ab8-4e49-9076-1517afb2299d
# ╠═93bca010-ef25-4672-a601-b42fbae73bd3
# ╠═987772c6-35e6-4340-b8ad-65eb6c5521dc
# ╠═ebc190c1-aa57-4ab0-8f6f-958eaae4008b
# ╠═892e2207-b7e0-410c-a22e-93aa185ea858
# ╠═97306279-df1f-4bb2-9d51-8e6345808100
# ╟─fb3490f1-0b3a-44b0-9bea-50cb5e530c15
# ╟─34d14afe-2194-43f0-a829-15276ecca883
# ╟─2ebc2b89-27f2-4abc-816e-7950bce2267f
# ╟─fb971894-a833-4955-93b0-da68c4a71db0
# ╟─f7451fdb-b5ac-4bf7-82f2-fed46ed98e99
# ╟─dac7e4e0-2532-45ff-8b0f-1d8a92e90ccf
# ╠═1679b86e-13ef-48c4-9a05-613387c213f9
# ╠═6e8805fe-05e8-49b5-a231-1efb995da173
# ╠═05cccc05-89a8-48c9-b1ac-a8ff6b5089a9
# ╟─1d3939e1-bcec-46fd-84e0-b84abfbb3dc9
