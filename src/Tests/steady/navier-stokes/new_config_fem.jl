include("../../../FEM/FEM.jl")

root = "/home/nicholasmueller/git_repos/Mabla.jl"
mesh_name = "cube5x5x5.json"
model = DiscreteModelFromFile(joinpath(root, "tests/meshes/$mesh_name"))

function set_labels!(bnd_info::Dict)
  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  @assert length(tags) == length(bnds)
  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels, tags[i], bnds[i])
    end
  end
end
bnd_info = Dict("dirichlet" => collect(1:25), "neumann" => [26])
set_labels!(bnd_info)

degree=2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γn = BoundaryTriangulation(model, tags=["neumann"])
dΓn = Measure(Γn, degree)

ranges = [[1., 10.], [1., 10.], [1., 10.],
          [1., 10.], [1., 10.], [1., 10.]]
param_space = ParamSpace(ranges,UniformSampling())

a(μ::Vector{Float},x) = 1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
a(μ::Vector{Float}) = x->a(μ,x)
b(μ::Vector{Float},x) = 1.
b(μ::Vector{Float}) = x->b(μ,x)
f(μ::Vector{Float},x) = 1. + Point(μ[4:6]) .* x
f(μ::Vector{Float}) = x->f(μ,x)
h(μ::Vector{Float},x) = 1. + Point(μ[4:6]) .* x
h(μ::Vector{Float}) = x->h(μ,x)
g(μ::Vector{Float},x) = 1. + Point(μ[4:6]) .* x
g(μ::Vector{Float}) = x->g(μ,x)

afe(μ,u,v) = ∫(a(μ) * ∇(v) ⊙ ∇(u))dΩ
bfe(μ,u,q) = ∫(b(μ) * q * (∇⋅(u)))dΩ
ffe(μ,v) = ∫(f(μ) ⋅ v)dΩ
hfe(μ,v) = ∫(h(μ) ⋅ v)dΓn
#= c(μ,u,v) = ∫(v ⊙ (∇(u)'⋅μ))dΩ
dc(μ,u,v) = ∫(v ⊙ (∇(μ)'⋅u))dΩ =#
conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
dc(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

rhs(μ,(v,q)) = ffe(μ,v) + hfe(μ,v)
lhs(μ,(u,p),(v,q)) = ∫(a(μ)*∇(v)⊙∇(u) - b(μ)*((∇⋅v)*p + q*(∇⋅u)))dΩ

res(μ,(u,p),(v,q)) = lhs(μ,(u,p),(v,q)) + c(u,v) - rhs(μ,(v,q))
jac(μ,(u,p),(du,dp),(v,q)) = lhs(μ,(du,dp),(v,q)) + dc(u,du,v)

reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},2)
reffe2 = Gridap.ReferenceFE(lagrangian,Float,1;space=:P)

I=true
S=true

Gμ = ParamFunctional(param_space,g;S)
myV = MyTests(model, reffe1; conformity=:H1, dirichlet_tags=["dirichlet"])
myU = MyTrials(myV,Gμ)
myQ = MyTests(model, reffe2; conformity=:L2)
myP = MyTrials(myQ)

myX = ParamMultiFieldTrialFESpace([myU.trial,myP.trial])
myY = MultiFieldFESpace([myV.test,myQ.test])
op = ParamFEOperator(res,jac,param_space,myX,myY)
nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
ye = solve(FESolver(nls),op,1)

opA = ParamVarOperator(a,afe,param_space,myU,myV)
opB = ParamVarOperator(b,bfe,param_space,myU,myQ)
opF = ParamVarOperator(f,ffe,param_space,myV)
opH = ParamVarOperator(h,hfe,param_space,myV)

stokes_problem = Problem(μ,[uh,ph],[opA,opB,opF,opH])

#= #COMPARISON
μ = realization(op)
V = TestFESpace(model, reffe1; conformity=:H1, dirichlet_tags=["dirichlet"])
U = TrialFESpace(V, g(μ))
P = TestFESpace(model, reffe2; conformity=:L2)
Q = TrialFESpace(P)
X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

xfun = FEFunction(X,ones(2930))
rx = assemble_vector(res(μ,xfun,get_fe_basis(Y)),Y)
nlop = ParamNonlinearOperator(
  get_algebraic_operator(op),xfun,μ,nothing)
emptyR = Gridap.ODEs.TransientFETools.allocate_residual(nlop,ones(2950))
rxnok = Gridap.ODEs.TransientFETools.residual!(emptyR,nlop,ones(2950)) =#



μ = realization(op)
V = TestFESpace(model, reffe1; conformity=:H1, dirichlet_tags=["dirichlet"])
U = TrialFESpace(V, g(μ))
P = TestFESpace(model, reffe2; conformity=:L2)
Q = TrialFESpace(P)
X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])
r((u,p),(v,q)) = res(μ,(u,p),(v,q))
j((u,p),(du,dp),(v,q)) = jac(μ,(u,p),(du,dp),(v,q))
operator = FEOperator(r,j,X,Y)
solve(FESolver(nls),operator)

a(u,v) = ∫(∇(v) ⊙ (a(μ) * ∇(u)))dΩ
b(u,q) = ∫(q*(∇⋅u))dΩ
rhs(v) = ∫(v ⋅ f(μ))dΩ + ∫(v ⋅ h(μ))dΓn
A = assemble_matrix(a,U,V)
B = assemble_matrix(b,U,Q)
triform1(u,v,z) = ∫(v ⊙ (∇(u)'⋅z))dΩ
triform1(z) = (u,v) -> triform1(u,v,z)
triform2(u,v,z) = ∫(v ⊙ (∇(z)'⋅u))dΩ
triform2(z) = (u,v) -> triform2(u,v,z)
C(z) = assemble_matrix(triform1(z),U,V)
D(z) = assemble_matrix(triform2(z),U,V)
J(z) = vcat(hcat(A+C(z)+D(z),-B'),hcat(B,zeros(500,500)))

FH = assemble_vector(rhs,V)
gd = interpolate_dirichlet(g(μ), U)
la(v) = ∫(∇(v) ⊙ (a(μ) * ∇(gd)))dΩ
lb(q) = ∫(q*(∇⋅gd))dΩ
LA = assemble_vector(la,V)
LB = assemble_vector(lb,Q)
Uall,Vall = myU.trial_no_bc,myV.test_no_bc
dir_dofs = myU.ddofs_on_full_trian
free_dofs = setdiff(collect(1:myU.trial_no_bc.nfree),myU.ddofs_on_full_trian)
Call(z) = assemble_matrix(triform1(z),Uall,Vall)
LC(z) = Call(z)[free_dofs,dir_dofs]*z.dirichlet_values
R(z) = vcat(FH-LA-LC(z),-LB)
