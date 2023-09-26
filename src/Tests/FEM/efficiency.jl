nsnaps = 10
μ = realization(feop,nsnaps)
sols = collect_solutions(fesolver,feop,uh0μ,μ,t0,tf)

list_tf = [0.05,0.25,0.5]
list_nμ = [10,50,100,200]
times,nallocs = [],[]
for tf in list_tf
  for nμ in list_nμ
    μ = realization(feop,nμ)
    result = @timed collect_solutions(fesolver,feop,uh0μ,μ,t0,tf)
    push!(times,result[:time])
    push!(nallocs,result[:bytes])
  end
end
#reference
@time collect_solutions(fesolver,feop,uh0μ,realization(feop,1),t0,dt)

tgroups = T[1:4],T[5:8],T[9:12]
nagroups = NA[1:4],NA[5:8],NA[9:12]
Nt = [10,50,100]
x = Nt*list_nμ'

scatter(x[1,:],tgroups[1])
scatter(x[1,:],nagroups[1])
