function poisson_functions(::Val{true},measures::ProblemFixedMeasures)

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)

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

function poisson_operators(
  measures::ProblemFixedMeasures,
  PS::ParamSpace,
  time_info::TimeInfo,
  V::FESpace,
  U::ParamTransientTrialFESpace;
  a::Function = (x,p::Param,t::Real)->1,
  m::Function = (x,p::Param,t::Real)->1,
  f::Function = (x,p::Param,t::Real)->VectorValue(0,0),
  h::Function = (x,p::Param,t::Real)->VectorValue(0,0))

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t)*∇(v)⊙∇(u))dΩ
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  afe(a,u,v) = ∫(a*∇(v)⊙∇(u))dΩ
  afe(a) = (u,v) -> afe(a,u,v)
  A = ParamFunctions(a,afe)

  mfe(p::Param,t::Real,dΩ,u,v) = ∫(m(p,t)*v⋅u)dΩ
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  mfe(m,u,v) = ∫(m*v⋅u)dΩ
  mfe(m) = (u,v) -> mfe(m,u,v)
  M = ParamFunctions(m,mfe)

  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t)⋅v)dΩ
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  ffe(f,v) = ∫(f⋅v)dΩ
  ffe(f) = v -> ffe(f,v)
  F = ParamFunctions(f,ffe)

  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t)⋅v)dΓn
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)
  hfe(h,v) = ∫(h⋅v)dΓn
  hfe(h) = v -> hfe(h,v)
  H = ParamFunctions(h,hfe)

  opA = NonaffineParamOperator(A,PS,time_info,U,V;id=:A)
  opM = AffineParamOperator(M,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(F,PS,time_info,V;id=:F)
  opH = AffineParamOperator(H,PS,time_info,V;id=:H)

  lhs_t(μ,t,u,v) = mfe(μ,t,u,v)
  lhs(p,t,u,v) = afe(p,t,u,v)
  rhs(p,t,v) = ffe(p,t,v) + hfe(p,t,v)

  feop = ParamTransientAffineFEOperator(lhs_t,lhs,rhs,PS,U,V)

  feop,opA,opM,opF,opH
end

function stokes_operators(measures::ProblemFixedMeasures)

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

function stokes_operators(
  measures::ProblemFixedMeasures,
  PS::ParamSpace,
  time_info::TimeInfo,
  V::FESpace,
  U::ParamTransientTrialFESpace,
  Q::FESpace,
  P::FESpace;
  a::Function = (x,p::Param,t::Real)->1,
  b::Function = (x,p::Param,t::Real)->1,
  m::Function = (x,p::Param,t::Real)->1,
  f::Function = (x,p::Param,t::Real)->VectorValue(0,0),
  h::Function = (x,p::Param,t::Real)->VectorValue(0,0))

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  Y = ParamTransientMultiFieldFESpace([V,Q])
  X = ParamTransientMultiFieldFESpace([U,P])

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t)*∇(v)⊙∇(u))dΩ
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  afe(a,u,v) = ∫(a*∇(v)⊙∇(u))dΩ
  afe(a) = (u,v) -> afe(a,u,v)
  A = ParamFunctions(a,afe)

  mfe(p::Param,t::Real,dΩ,u,v) = ∫(m(p,t)*v⋅u)dΩ
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  mfe(m,u,v) = ∫(m*v⋅u)dΩ
  mfe(m) = (u,v) -> mfe(m,u,v)
  M = ParamFunctions(m,mfe)

  bfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t)*q*(∇⋅(u)))dΩ
  bfe(p::Param,t::Real,u,q) = bfe(p,t,dΩ,u,q)
  bfe(p::Param,t::Real) = (u,q) -> bfe(p,t,u,q)
  bfe(b,u,q) = ∫(b*q*(∇⋅(u)))dΩ
  bfe(b) = (u,q) -> bfe(b,u,q)
  B = ParamFunctions(b,bfe)

  btfe(p::Param,t::Real,dΩ,u,v) = ∫(b(p,t)*(∇⋅(v))*u)dΩ
  btfe(p::Param,t::Real,u,v) = btfe(p,t,dΩ,u,v)
  btfe(p::Param,t::Real) = (u,v) -> btfe(p,t,u,v)
  btfe(b,u,v) = ∫(b*(∇⋅(v))*u)dΩ
  btfe(b) = (u,v) -> btfe(b,u,v)
  BT = ParamFunctions(b,btfe)

  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t)⋅v)dΩ
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  ffe(f,v) = ∫(f⋅v)dΩ
  ffe(f) = v -> ffe(f,v)
  F = ParamFunctions(f,ffe)

  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t)⋅v)dΓn
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)
  hfe(h,v) = ∫(h⋅v)dΓn
  hfe(h) = v -> hfe(h,v)
  H = ParamFunctions(h,hfe)

  opA = NonaffineParamOperator(A,PS,time_info,U,V;id=:A)
  opB = AffineParamOperator(B,PS,time_info,U,Q;id=:B)
  opBT = AffineParamOperator(BT,PS,time_info,P,V;id=:BT)
  opM = AffineParamOperator(M,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(F,PS,time_info,V;id=:F)
  opH = AffineParamOperator(H,PS,time_info,V;id=:H)

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)
  lhs_t(μ,t,(u,p),(v,q)) = mfe(μ,t,u,v)

  feop = ParamTransientAffineFEOperator(lhs_t,lhs,rhs,PS,X,Y)

  feop,opA,opB,opBT,opM,opF,opH
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

function navier_stokes_functions(
  measures::ProblemFixedMeasures,
  PS::ParamSpace,
  time_info::TimeInfo,
  V::FESpace,
  U::ParamTransientTrialFESpace,
  Q::FESpace,
  P::FESpace;
  a::Function = (x,p::Param,t::Real)->1,
  b::Function = (x,p::Param,t::Real)->1,
  c::Function = (x,p::Param,t::Real)->1,
  d::Function = (x,p::Param,t::Real)->1,
  m::Function = (x,p::Param,t::Real)->1,
  f::Function = (x,p::Param,t::Real)->VectorValue(0,0),
  h::Function = (x,p::Param,t::Real)->VectorValue(0,0))

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  Y = ParamTransientMultiFieldFESpace([V,Q])
  X = ParamTransientMultiFieldFESpace([U,P])

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t)*∇(v)⊙∇(u))dΩ
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  afe(a::Function,u,v) = ∫(a*∇(v)⊙∇(u))dΩ
  afe(a::Function) = (u,v) -> afe(a,u,v)
  A = ParamFunctions(a,afe)

  mfe(p::Param,t::Real,dΩ,u,v) = ∫(m(p,t)*v⋅u)dΩ
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  mfe(m::Function,u,v) = ∫(m*v⋅u)dΩ
  mfe(m::Function) = (u,v) -> mfe(m,u,v)
  M = ParamFunctions(m,mfe)

  bfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t)*q*(∇⋅(u)))dΩ
  bfe(p::Param,t::Real,u,q) = bfe(p,t,dΩ,u,q)
  bfe(p::Param,t::Real) = (u,q) -> bfe(p,t,u,q)
  bfe(b::Function,u,q) = ∫(b*q*(∇⋅(u)))dΩ
  bfe(b::Function) = (u,q) -> bfe(b,u,q)
  B = ParamFunctions(b,bfe)

  btfe(p::Param,t::Real,dΩ,u,v) = ∫(b(p,t)*(∇⋅(v))*u)dΩ
  btfe(p::Param,t::Real,u,v) = btfe(p,t,dΩ,u,v)
  btfe(p::Param,t::Real) = (u,v) -> btfe(p,t,u,v)
  btfe(b::Function,u,v) = ∫(b*(∇⋅(v))*u)dΩ
  btfe(b::Function) = (u,v) -> btfe(b,u,v)
  BT = ParamFunctions(b,btfe)

  cfe(dΩ,z,u,v) = ∫(v⊙(∇(u)'⋅z))dΩ
  cfe(z,u,v) = cfe(dΩ,z,u,v)
  cfe(z) = (u,v) -> cfe(z,u,v)
  C = ParamFunctions(c,cfe)

  dfe(dΩ,z,u,v) = ∫(v⊙(∇(z)'⋅u))dΩ
  dfe(z,u,v) = dfe(dΩ,z,u,v)
  dfe(z) = (u,v) -> dfe(z,u,v)
  D = ParamFunctions(d,dfe)

  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t)⋅v)dΩ
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  ffe(f::Function,v) = ∫(f⋅v)dΩ
  ffe(f::Function) = v -> ffe(f,v)
  F = ParamFunctions(f,ffe)

  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t)⋅v)dΓn
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)
  hfe(h::Function,v) = ∫(h⋅v)dΓn
  hfe(h::Function) = v -> hfe(h,v)
  H = ParamFunctions(h,hfe)

  opA = AffineParamOperator(A,PS,time_info,U,V;id=:A)
  opB = AffineParamOperator(B,PS,time_info,U,Q;id=:B)
  opBT = AffineParamOperator(BT,PS,time_info,P,V;id=:BT)
  opC = NonlinearParamOperator(C,PS,time_info,U,V;id=:C)
  opD = NonlinearParamOperator(D,PS,time_info,U,V;id=:D)
  opM = AffineParamOperator(M,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(F,PS,time_info,V;id=:F)
  opH = AffineParamOperator(H,PS,time_info,V;id=:H)

  # BETTER CONVERGENCE WITH THIS DEF OF NONLINEAR TERM
  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  cgridap(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dcgridap(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)

  res(μ,t,(u,p),(v,q)) = mfe(μ,t,∂t(u),v) + lhs(μ,t,(u,p),(v,q)) + cgridap(u,v) - rhs(μ,t,(v,q))
  jac(μ,t,(u,p),(du,dp),(v,q)) = lhs(μ,t,(du,dp),(v,q)) + dcgridap(u,du,v)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = mfe(μ,t,dut,v)

  feop = ParamTransientFEOperator(res,jac,jac_t,PS,X,Y)

  feop,opA,opB,opBT,opC,opD,opM,opF,opH
end
