function poisson_functions(
  dΩ::Measure,
  dΓn::Measure,
  ::Val{true})

  function a(x,p::Param)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(p::Param) = x->a(p,x)
  function f(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  f(p::Param) = x->f(p,x)
  function h(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  h(p::Param) = x->h(p,x)
  function g(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  g(p::Param) = x->g(p,x)

  afe(p,u,v) = ∫(a(p) * ∇(v) ⋅ ∇(u))dΩ
  ffe(p,v) = ∫(f(p) * v)dΩ
  hfe(p,v) = ∫(h(p) * v)dΓn

  lhs(p,u,v) = afe(p,u,v)
  rhs(p,v) = ffe(p,v) + hfe(p,v)

  a,afe,f,ffe,h,hfe,g,lhs,rhs
end

function navier_stokes_functions(
  dΩ::Measure,
  dΓn::Measure,
  P::ParamSpace,
  ::Val{true})

  function a(x,p::Param)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(μ::Param) = x->a(μ,x)
  b(μ::Param,x) = 1.
  b(μ::Param) = x->b(μ,x)
  function f(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  f(μ::Param) = x->f(μ,x)
  function h(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  h(μ::Param) = x->h(μ,x)
  function g(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  g(μ::Param) = x->g(μ,x)

  afe(μ,u,v) = ∫(a(μ) * ∇(v) ⊙ ∇(u))dΩ
  bfe(μ,u,q) = ∫(b(μ) * q * (∇⋅(u)))dΩ
  cfe(μ,z,u,v) = ∫(c(μ)*v⊙(∇(u)'⋅z))dΩ
  cfe(μ,z) = (u,v) -> cfe(μ,z,u,v)
  cfe(z) = cfe(realization(P),z)
  dfe(μ,z,u,v) = ∫(d(μ)*v⊙(∇(z)'⋅u))dΩ
  dfe(μ,z) = (u,v) -> dfe(μ,z,u,v)
  dfe(z) = dfe(realization(P),z)
  ffe(μ,v) = ∫(f(μ) ⋅ v)dΩ
  hfe(μ,v) = ∫(h(μ) ⋅ v)dΓn

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  cgridap(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dcgridap(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,(v,q)) = ffe(μ,v) + hfe(μ,v)
  lhs(μ,(u,p),(v,q)) = afe(μ,u,v) - bfe(μ,v,p) - bfe(μ,u,q)

  res(μ,(u,p),(v,q)) = lhs(μ,(u,p),(v,q)) + c(u,v) - rhs(μ,(v,q))
  jac(μ,(u,p),(du,dp),(v,q)) = lhs(μ,(du,dp),(v,q)) + dc(u,du,v)

  afe,bfe,cfe,dfe,ffe,hfe,aμ,bμ,cμ,dμ,fμ,hμ,gμ,res,jac
end

function navier_stokes_functions(
  dΩ::Measure,
  dΓn::Measure,
  P::ParamSpace,
  ::Val{false})

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(μ::Param,t::Real) = x->a(x,μ,t)
  a(μ::Param) = t->a(μ,t)
  aμ = ParamFunction(ptype,:A,P,a)
  b(x,μ::Param,t::Real) = 1.
  b(μ::Param,t::Real) = x->b(x,μ,t)
  b(μ::Param) = t->b(μ,t)
  bμ = ParamFunction(ptype,:B,P,b)
  c(x,μ::Param,t::Real) = 1.
  c(μ::Param,t::Real) = x->c(x,μ,t)
  c(μ::Param) = t->c(μ,t)
  cμ = ParamFunction(ptype,:C,P,c)
  d(x,μ::Param,t::Real) = 1.
  d(μ::Param,t::Real) = x->d(x,μ,t)
  d(μ::Param) = t->d(μ,t)
  dμ = ParamFunction(ptype,:D,P,d)
  function f(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sin(t)*Point(μ[4:6]).*x
  end
  f(μ::Param,t::Real) = x->f(x,μ,t)
  f(μ::Param) = t->f(μ,t)
  fμ = ParamFunction(ptype,:F,P,f)
  function h(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sin(t)*Point(μ[4:6]).*x
  end
  h(μ::Param,t::Real) = x->h(x,μ,t)
  h(μ::Param) = t->h(μ,t)
  hμ = ParamFunction(ptype,:H,P,h)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sin(t)*Point(μ[4:6]).*x
  end
  g(μ::Param,t::Real) = x->g(x,μ,t)
  g(μ::Param) = t->g(μ,t)
  gμ = ParamFunction(ptype,:G,P,g)
  m(x,μ::Param,t::Real) = 1.
  m(μ::Param,t::Real) = x->m(x,μ,t)
  m(μ::Param) = t->m(μ,t)
  mμ = ParamFunction(ptype,:M,P,m)

  mfe(μ,t,u,v) = ∫(v⋅u)dΩ
  afe(μ,t,u,v) = ∫(a(μ,t)*∇(v) ⊙ ∇(u))dΩ
  bfe(μ,t,u,q) = ∫(b(μ,t)*q*(∇⋅(u)))dΩ
  cfe(μ,t,z,u,v) = ∫(c(μ,t)*v⊙(∇(u)'⋅z))dΩ
  cfe(μ,t,z) = (u,v) -> cfe(μ,t,z,u,v)
  cfe(z) = cfe(realization(P),0.,z)
  dfe(μ,t,z,u,v) = ∫(d(μ,t)*v⊙(∇(z)'⋅u))dΩ
  dfe(μ,t,z) = (u,v) -> dfe(μ,t,z,u,v)
  dfe(z) = dfe(realization(P),0.,z)
  ffe(μ,t,v) = ∫(f(μ,t)⋅v)dΩ
  hfe(μ,t,v) = ∫(h(μ,t)⋅v)dΓn
  #FOR SOME REASON, BETTER CONVERGENCE WITH THIS DEF OF NONLINEAR TERM
  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  cgridap(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dcgridap(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)

  res(μ,t,(u,p),(v,q)) = mfe(μ,t,(∂t(u),∂t(p)),(v,q)) + lhs(μ,t,(u,p),(v,q)) + cgridap(u,v) - rhs(μ,t,(v,q))
  jac(μ,t,(u,p),(du,dp),(v,q)) = lhs(μ,t,(du,dp),(v,q)) + dcgridap(u,du,v)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = mfe(μ,t,(dut,dpt),(v,q))

  afe,bfe,cfe,dfe,ffe,hfe,mfe,aμ,bμ,cμ,dμ,fμ,hμ,gμ,mμ,res,jac,jac_t
end
