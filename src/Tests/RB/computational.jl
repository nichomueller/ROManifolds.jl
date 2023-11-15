μ = [rand(3) for _ = 1:10]
t = rand(10)
@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))dΩ
@time ∫(aμt(μ,t)*∇(_dv)⋅∇(_du))_dΩ
@time ∫(aμt(μ,t)*∇(__dv)⋅∇(__du))__dΩ
@time ∫(a(μ[1],t[1])*∇(dv)⋅∇(du))dΩ

rtrian = view(Ω,idx)
rmeas = Measure(rtrian,2)
@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))rmeas


rmodel = Geometry.DiscreteModelPortion(model,cell_to_parent_cell)
rtest = TestFESpace(rmodel,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
