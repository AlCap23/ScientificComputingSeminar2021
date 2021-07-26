using LinearAlgebra: length
using LinearAlgebra

g(x, y) = x + sin(x)*y^2

function g(x::Vector, y::Vector) 
    z = similar(x)
    for i in 1:length(x)
        z[i] = g(x[i], y[i])
    end
    return z
end

x = randn(10); y = randn(10);

@time g(x,y) 

@time g(x,y) 

# Use another input vector of type Float32
z = randn(Float32, 10);

@time g(z,y)

# Benchmark further
using BenchmarkTools

function sum_vectors!(x, y, z)
    n = length(x)
    for i in 1:n
        x[i] = y[i] + z[i]
    end
end

function sum_vectors_simd!(x, y, z)
    n = length(x)
    @inbounds @simd for i in 1:n
        x[i] = y[i] + z[i]
    end
end

a = zeros(Float32, 100_000);
b = randn(Float32, 100_000);
c = randn(Float32, 100_000);

@btime sum_vectors!($a,$b,$c)
@btime sum_vectors_simd!($a,$b,$c)

@code_llvm sum_vectors_simd!(a,b,c)