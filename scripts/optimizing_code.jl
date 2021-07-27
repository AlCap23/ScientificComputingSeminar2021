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

## Memory Layout
using BenchmarkTools

function sum_by_row(x::AbstractMatrix)
    res = zero(eltype(x))
    for i in 1:size(x, 1)
        res += sum(x[i,:])
    end
    return res
end

function sum_by_col(x::AbstractMatrix)
    res = zero(eltype(x))
    for i in 1:size(x, 2)
        res += sum(x[:,i])
    end
    return res
end

x = randn(Float32, 10_000, 10_000);

@btime sum_by_row($x) # 713.33 ms (20000 allocations: 382.23 MiB)
@btime sum_by_col($x) # 99.842 ms (20000 allocations: 382.23 MiB)

## SIMD
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

@btime sum_vectors!($a,$b,$c) # 79.932 μs (0 allocations: 0 bytes)
@btime sum_vectors_simd!($a,$b,$c) # 18.960 μs (0 allocations: 0 bytes)

## For the interested
@code_llvm sum_vectors_simd!(a,b,c)

## Multithreading
using Base.Threads

function sum_vectors_threading!(x, y, z)
    n = length(x)
    @inbounds Threads.@threads for i in 1:n
        x[i] = y[i] + z[i]
    end
end

@btime sum_vectors_threading!($a,$b,$c) # 132.019 μs (6 allocations: 560 bytes)