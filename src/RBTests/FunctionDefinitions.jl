function poisson_functions(ptype::ProblemType,measures::ProblemMeasures)
  poisson_functions(issteady(ptype),measures)
end

function poisson_functions(::Val{true},measures::ProblemFixedMeasures)

  function a(x,p::Param)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5]*exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(p::Param) = x->a(x,p)
  function f(x,p::Param)
    1.
  end
  f(p::Param) = x->f(x,p)
  function h(x,p::Param)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5]*exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  h(p::Param) = x->h(x,p)
  function g(x,p::Param)
    μ = get_μ(p)
    (1. + cos(norm(Point(μ[1:3]).*x))) / norm(1. + cos(norm(Point(μ[1:3]).*x)))
  end
  g(p::Param) = x->g(x,p)

  afe(p::Param,dΩ,u,v) = ∫(a(p)*∇(v)⋅∇(u))dΩ
  ffe(p::Param,dΩ,v) = ∫(f(p)*v)dΩ
  hfe(p::Param,dΓn,v) = ∫(h(p)*v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,u,v) = afe(p,dΩ,u,v)
  afe(p::Param) = (u,v) -> afe(p,u,v)
  ffe(p::Param,v) = ffe(p,dΩ,v)
  ffe(p::Param) = v -> ffe(p,v)
  hfe(p::Param,v) = hfe(p,dΓn,v)
  hfe(p::Param) = v -> hfe(p,v)

  lhs(p,u,v) = afe(p,u,v)
  rhs(p,v) = ffe(p,v) + hfe(p,v)

  a,afe,f,ffe,h,hfe,g,lhs,rhs
end

function poisson_functions(::Val{false},measures::ProblemFixedMeasures)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    if (x[1] ≤ 0.75 && x[2] ≤ 0.5)
      μ[1]
    elseif (x[1] ≤ 0.75 && x[2] > 0.5)
      μ[2]
    elseif (x[1] > 0.75 && x[2] ≤ 0.5)
      μ[3]
    else
      μ[4]
    end
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  a(p::Param) = t->a(p,t)
  function m(x,p::Param,t::Real)
    1.
  end
  m(p::Param,t::Real) = x->m(x,p,t)
  m(p::Param) = t->m(p,t)
  function f(x,p::Param,t::Real)
    μ = get_μ(p)
    1.
  end
  f(p::Param,t::Real) = x->f(x,p,t)
  f(p::Param) = t->f(p,t)
  function h(x,p::Param,t::Real)
    μ = get_μ(p)
    1.
  end
  h(p::Param,t::Real) = x->h(x,p,t)
  h(p::Param) = t->h(p,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    exp(-x[1]/μ[2])*abs.(sin(μ[3]*t))*minimum(μ)
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g(p::Param) = t->g(p,t)

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
  mfe(p::Param,t::Real,dΩ,u,v) = ∫(m(p,t)*v⋅u)dΩ
  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t)*v)dΩ
  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t)*v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)
  # FUNCTIONAL MDEIM REPRESENTATION
  afe(a,u,v) = ∫(a*∇(v)⋅∇(u))dΩ
  mfe(m,u,v) = ∫(m*v⋅u)dΩ
  ffe(f,v) = ∫(f*v)dΩ
  hfe(h,v) = ∫(h*v)dΓn

  lhs(p,t,u,v) = afe(p,t,u,v)
  rhs(p,t,v) = ffe(p,t,v) + hfe(p,t,v)

  a,afe,m,mfe,f,ffe,h,hfe,g,lhs,rhs
end

function stokes_functions(ptype::ProblemType,measures::ProblemMeasures)
  stokes_functions(issteady(ptype),measures)
end

function stokes_functions(::Val{true},measures::ProblemFixedMeasures)

  function a(x,p::Param)
    μ = get_μ(p)
    5*μ[1] + abs(sin(norm(x.*Point(μ[1:3])))*μ[4])
  end
  a(p::Param) = x->a(x,p)
  b(x,p::Param) = 1.
  b(p::Param) = x->b(x,p)
  f(x,p::Param) = VectorValue(0.,0.,0.)
  f(p::Param) = x->f(x,p)
  h(x,p::Param) = VectorValue(0.,0.,0.)
  h(p::Param) = x->h(x,p)
  function g(x,p::Param)
    μ = get_μ(p)
    R = 0.5
    dist = sum(x^2)/(R^2)
    abs(sin(μ[1]))*VectorValue(0.,0.,1-dist)*(x[3]==0.)
  end
  g(p::Param) = x->g(x,p)

  afe(p::Param,dΩ,u,v) = ∫(a(p)*∇(v)⊙∇(u))dΩ
  bfe(p::Param,dΩ,u,q) = ∫(b(p)*q*(∇⋅(u)))dΩ
  ffe(p::Param,dΩ,v) = ∫(f(p)⋅v)dΩ
  hfe(p::Param,dΓn,v) = ∫(h(p)⋅v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,u,v) = afe(p,dΩ,u,v)
  afe(p::Param) = (u,v) -> afe(p,u,v)
  bfe(p::Param,u,v) = bfe(p,dΩ,u,v)
  bfe(p::Param) = (u,v) -> bfe(p,u,v)
  ffe(p::Param,v) = ffe(p,dΩ,v)
  ffe(p::Param) = v -> ffe(p,v)
  hfe(p::Param,v) = hfe(p,dΓn,v)
  hfe(p::Param) = v -> hfe(p,v)

  rhs(μ,(v,q)) = ffe(μ,v) + hfe(μ,v)
  lhs(μ,(u,p),(v,q)) = afe(μ,u,v) - bfe(μ,v,p) - bfe(μ,u,q)

  a,afe,b,bfe,f,ffe,h,hfe,g,lhs,rhs
end

function stokes_functions(::Val{false},measures::ProblemFixedMeasures)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1/sum(μ)
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  a(p::Param) = t->a(p,t)
  m(x,p::Param,t::Real) = 1.
  m(p::Param,t::Real) = x->m(x,p,t)
  m(p::Param) = t->m(p,t)
  b(x,p::Param,t::Real) = 1.
  b(p::Param,t::Real) = x->b(x,p,t)
  b(p::Param) = t->b(p,t)
  f(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  f(p::Param,t::Real) = x->f(x,p,t)
  h(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  h(p::Param,t::Real) = x->h(x,p,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    R = 0.5
    T = 2.5
    dist = (x[1]^2+x[2]^2)/(R^2)
    abs(1-cos(2*pi*t/T)+sin(μ[1]*2*pi*t/T)/μ[2])*VectorValue(0.,0.,1-dist)*(x[3]==0.)
  end
  g(p::Param,t::Real) = x->g(x,p,t)

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t)*∇(v)⊙∇(u))dΩ
  mfe(p::Param,t::Real,dΩ,u,v) = ∫(v⋅u)dΩ
  bfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t)*q*(∇⋅(u)))dΩ
  bTfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t)*(∇⋅(q))*u)dΩ
  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t)⋅v)dΩ
  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t)⋅v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  bfe(p::Param,t::Real,u,v) = bfe(p,t,dΩ,u,v)
  bfe(p::Param,t::Real) = (u,v) -> bfe(p,t,u,v)
  bTfe(p::Param,t::Real,u,v) = bTfe(p,t,dΩ,u,v)
  bTfe(p::Param,t::Real) = (u,v) -> bTfe(p,t,u,v)
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  mfe_gridap(μ,t,(u,p),(v,q)) = mfe(μ,t,u,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)

  a,afe,m,mfe,mfe_gridap,b,bfe,bTfe,f,ffe,h,hfe,g,lhs,rhs
end

function navier_stokes_functions(ptype::ProblemType,measures::ProblemMeasures)
  navier_stokes_functions(issteady(ptype),measures)
end

function navier_stokes_functions(::Val{true},measures::ProblemFixedMeasures)

  function a(x,p::Param)
    μ = get_μ(p)
    5*μ[1] + abs(sin(norm(x.*Point(μ[1:3])))*μ[4])
  end
  a(p::Param) = x->a(x,p)
  b(x,p::Param) = 1.
  b(p::Param) = x->b(x,p)
  c(x,p::Param) = 1.
  c(p::Param) = x->c(x,p)
  d(x,p::Param) = 1.
  d(p::Param) = x->d(x,p)
  f(x,p::Param) = VectorValue(0.,0.,0.)
  f(p::Param) = x->f(x,p)
  h(x,p::Param) = VectorValue(0.,0.,0.)
  h(p::Param) = x->h(x,p)
  function g(x,p::Param)
    μ = get_μ(p)
    R = 0.5
    dist = sum(x^2)/(R^2)
    μ[1]*VectorValue(0.,0.,1-dist)*(x[3]==0.)
  end
  g(p::Param) = x->g(x,p)

  afe(p::Param,dΩ,u,v) = ∫(a(p)*∇(v)⊙∇(u))dΩ
  bfe(p::Param,dΩ,u,q) = ∫(b(p)*q*(∇⋅(u)))dΩ
  cfe(dΩ,z,u,v) = ∫(v⊙(∇(u)'⋅z))dΩ
  dfe(dΩ,z,u,v) = ∫(v⊙(∇(z)'⋅u))dΩ
  ffe(p::Param,dΩ,v) = ∫(f(p)⋅v)dΩ
  hfe(p::Param,dΓn,v) = ∫(h(p)⋅v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,u,v) = afe(p,dΩ,u,v)
  afe(p::Param) = (u,v) -> afe(p,u,v)
  bfe(p::Param,u,v) = bfe(p,dΩ,u,v)
  bfe(p::Param) = (u,v) -> bfe(p,u,v)
  cfe(z,u,v) = cfe(dΩ,z,u,v)
  cfe(z) = (u,v) -> cfe(z,u,v)
  dfe(z,u,v) = dfe(dΩ,z,u,v)
  dfe(z) = (u,v) -> dfe(z,u,v)
  ffe(p::Param,v) = ffe(p,dΩ,v)
  ffe(p::Param) = v -> ffe(p,v)
  hfe(p::Param,v) = hfe(p,dΓn,v)
  hfe(p::Param) = v -> hfe(p,v)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  cgridap(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dcgridap(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,(v,q)) = ffe(μ,v) + hfe(μ,v)
  lhs(μ,(u,p),(v,q)) = afe(μ,u,v) - bfe(μ,v,p) - bfe(μ,u,q)

  res(μ,(u,p),(v,q)) = lhs(μ,(u,p),(v,q)) + cgridap(u,v) - rhs(μ,(v,q))
  jac(μ,(u,p),(du,dp),(v,q)) = lhs(μ,(du,dp),(v,q)) + dcgridap(u,du,v)

  a,afe,b,bfe,c,cfe,d,dfe,f,ffe,h,hfe,g,res,jac
end

function navier_stokes_functions(::Val{false},measures::ProblemFixedMeasures)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1e-2*μ[1]
  end
  a(μ::Param,t::Real) = x->a(x,μ,t)

  b(x,μ::Param,t::Real) = 1.
  b(μ::Param,t::Real) = x->b(x,μ,t)

  c(x,μ::Param,t::Real) = 1.
  c(μ::Param,t::Real) = x->c(x,μ,t)

  d(x,μ::Param,t::Real) = 1.
  d(μ::Param,t::Real) = x->d(x,μ,t)

  f(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  f(μ::Param,t::Real) = x->f(x,μ,t)

  h(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  h(μ::Param,t::Real) = x->h(x,μ,t)

  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    R = 0.5
    dist = (x[1]^2+x[2]^2)/(R^2)
    abs(1-cos(t)+μ[2]*sin(μ[3]*t))*VectorValue(0.,0.,1-dist)*(x[3]==0.)
  end
  g(μ::Param,t::Real) = x->g(x,μ,t)

  m(x,μ::Param,t::Real) = 1.
  m(μ::Param,t::Real) = x->m(x,μ,t)

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t)*∇(v)⊙∇(u))dΩ
  mfe(p::Param,t::Real,dΩ,u,v) = ∫(v⋅u)dΩ
  bfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t)*q*(∇⋅(u)))dΩ
  bTfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t)*(∇⋅(q))*u)dΩ
  cfe(dΩ,z,u,v) = ∫(v⊙(∇(u)'⋅z))dΩ
  dfe(dΩ,z,u,v) = ∫(v⊙(∇(z)'⋅u))dΩ
  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t)⋅v)dΩ
  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t)⋅v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  bfe(p::Param,t::Real,u,v) = bfe(p,t,dΩ,u,v)
  bfe(p::Param,t::Real) = (u,v) -> bfe(p,t,u,v)
  bTfe(p::Param,t::Real,u,v) = bTfe(p,t,dΩ,u,v)
  bTfe(p::Param,t::Real) = (u,v) -> bTfe(p,t,u,v)
  cfe(z,u,v) = cfe(dΩ,z,u,v)
  cfe(z) = (u,v) -> cfe(z,u,v)
  dfe(z,u,v) = dfe(dΩ,z,u,v)
  dfe(z) = (u,v) -> dfe(z,u,v)
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)

  # FUNCTIONAL MDEIM REPRESENTATION
  afe(a,u,v) = ∫(a*∇(v)⊙∇(u))dΩ
  bfe(b,u,q) = ∫(b*q*(∇⋅(u)))dΩ
  bTfe(b,u,q) = ∫(b*(∇⋅(q))*u)dΩ
  mfe(m,u,v) = ∫(m*v⋅u)dΩ
  ffe(f,v) = ∫(f*v)dΩ
  hfe(h,v) = ∫(h*v)dΓn

  #FOR SOME REASON, BETTER CONVERGENCE WITH THIS DEF OF NONLINEAR TERM
  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  cgridap(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dcgridap(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)

  res(μ,t,(u,p),(v,q)) = mfe(μ,t,∂t(u),v) + lhs(μ,t,(u,p),(v,q)) + cgridap(u,v) - rhs(μ,t,(v,q))
  jac(μ,t,(u,p),(du,dp),(v,q)) = lhs(μ,t,(du,dp),(v,q)) + dcgridap(u,du,v)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = mfe(μ,t,dut,v)

  a,afe,m,mfe,b,bfe,bTfe,c,cfe,d,dfe,f,ffe,h,hfe,g,res,jac,jac_t
end
