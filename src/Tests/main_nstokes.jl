a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

f(x,μ,t) = VectorValue(0,0)
f(μ,t) = x->f(x,μ,t)

h(x,μ,t) = aVectorValue(0,0)
h(μ,t) = x->h(x,μ,t)

g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
g(μ,t) = x->g(x,μ,t)
g0(x,μ,t) = VectorValue(0,0)
g0(μ,t) = x->g0(x,μ,t)

u0(x,μ) = VectorValue(0,0)
u0(μ) = x->u0(x,μ)
p0(x,μ) = 0
p0(μ) = x->p0(x,μ)

lhs(μ,t,(u,p),(du,dp),(dv,dq),dΩ) = (∫(dv⋅∂ₚt(du))dΩ + ∫(a(μ,t)*∇(dv)⊙∇(du))dΩ +
  ∫(dv⊙(dconv∘(du,∇(du),u,∇(u))))dΩ - ∫(dp*(∇⋅(dv)))dΩ - ∫(dq*(∇⋅(du)))dΩ)
rhs(μ,t,(dv,dq),dΩ) = ∫(f(μ,t)*dv)dΩ + ∫(h(μ,t)*dv)dΓn
djac_dt(μ,t,(u,p),(dut,dpt),(dv,dq),dΩ) = ∫(dv⋅dut)dΩ
