# first, define your residual

# u(m)   # solve a (non)linear PDE problem R(m,u(m)) = 0

# R(m,u) # where u is st R(m,u(m)) = 0 (implicit function theorem provides u(m))

# R(m) = m -> R(m,u(m))

# ∂ₘR(m₀,u₀) # partial derivative of R wrt m at m₀, u₀ = u(m₀)

# ∂ᵤR(m₀,u₀) # partial derivative of R wrt u at m₀, u₀ = u(m₀)

# Dₘu(m₀)    # total derivative of u wrt m at m₀

# Implicit function theorem provides
# DₘR(m) = ∂ₘR(m,u(m)) + ∂ᵤR(m,u(m)) Dₘu(m) = 0
# Dₘu(m) = inv(∂ᵤR(m,u(m))) * ∂ₘR(m,u(m))

# Rₐ # adjoint for R at R(m₀)

# Rₐ_Dₘu(Rₐ) = Rₐ -> (Rₐ * inv(∂ᵤR(m₀,u₀))) * ∂ₘR(m₀,u₀) # u₀ = u(m₀)

# We have a cost function in terms of m and u(m)

# L(m) = f(m,u(m))

# We can use an AD machinery for this problem. We requires a rule for u(m)

# Simplest test

# f(m) = l(m,u(m))
# l(m,u) = (m^2 + 10*u)^2
# u(m) s.t. r(u,m) = u*10 - m = 0
# u(m) = m/10, thus, f(m) = (m^2+m)^2, arg min f(m) = 0

# Use ChainRules.jl to define a reverse rule for u using implicit function theorem
# Use the rule within a Reverse AD package

using ChainRulesCore

# f(m) = l(m,u)

u(m) = m/10

gradient(u,1.0)[1] ≈ 0.1 # 1/10

import ChainRulesCore.rrule

function ChainRulesCore.rrule(::typeof(u), m)
    output = u(m)
    function u_pullback(yadj)
        g = yadj/10
        return (NoTangent(), g)
    end
    return output, u_pullback
end

gradient(foo2,1.0)[1]

using ChainRulesTestUtils

test_rrule(u, 0.5)

g(m,u) = (m^2 + 10*u)^2

function foo2(m)
    g_mu = g(m,u(m)) # g =  (m^2+m)^2
    return g_mu
end

using Zygote

gradient(foo2,1.0)[1] ≈ 12.0 # 2*(m^2+m)(2*m+1), internally using the rule

# A = rand(10,10)

using LinearAlgebra
A = diagm([1.0,2.0,3.0,4.0])
inv(A)

x = rand(4)
sol(x) = inv(A)*x
loss(x) = x'*sol(x)

y = A\x
A*y ≈ x

function ChainRulesCore.rrule(::typeof(sol), x::AbstractVector)
    output = inv(A)*x
    function sol_pullback(yadj)
        # g = inv(A)'*yadj
        g = A \ yadj # g' = y' * inv(A); inv(A)'*g = y;
        return (NoTangent(), g)
    end
    return output, sol_pullback
end

function ChainRulesCore.frule(::typeof(sol), x::AbstractVector)
    println("You don't want to use forward mode here")
    return nothing
end

# I don't understand why it does not work
# test_rrule(sol, x)

gradient(loss,x)[1] ≈ 2*(x'*inv(A))' # last ' because it is the gradient
