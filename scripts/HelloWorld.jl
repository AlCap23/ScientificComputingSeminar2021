# Define a function 
f(x,y) = exp(-(x-y)^2)

# Evaluate the function
f(0.2, 1) # Returns 0.5272924240430484

# Evaluate with a Float32
f(0.2f0, 1) #Returns 0.5272924f0

@code_warntype f(0.2, 1)

f(im*0.2, 1f0) # 0.3526677021528701 + 0.1491055129382033im

# Infer which function is called
@which exp(0.2f0)
@which exp(im*0.2f0)

# Simple definition of a data structure
using Base: Number
struct MyNumber <: Number end

# Dispatch on the definition
Base.exp(t::MyNumber) = println("Foo")

# Execute
a = MyNumber()
exp(a) # Returns "Foo"

# Import Linear Algebra
using LinearAlgebra

# Add a function dispatch on two vectors
f(x::AbstractVector, y::AbstractVector) = exp.(-(x-y).^2)

# Define two random vectors
x = randn(10); y = randn(10);

f(x,y) # Returns a vector

methods(f) # Show the available methods corresponding to f