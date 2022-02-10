module CnnInterpolation

using Optim
using Flux
using ChainRulesCore, ChainRulesTestUtils
using Gridap
using ReverseDiff, Zygote 

# Background model
L=1
domain = (0,L,0,L)
n_cells = 15
cells=(n_cells,n_cells)
h=L/n_cells
bgmodel = CartesianDiscreteModel(domain,cells)
order = 1
Ωbg = Triangulation(bgmodel)
Vbg = TestFESpace(bgmodel,FiniteElements(PhysicalDomain(),bgmodel,lagrangian,Float32,order))

# Target function to interploate
Nₓ(x) = x[1]^2 + x[2]^2
NTₕ = interpolate(Nₓ,Vbg)
NT = get_free_dof_values(NTₕ)

# Defining the neural network

# This network is comprised of a few dense layers followed by the decoder of a U-net. 
function normalisation(x)
	mean = Statistics.mean([1])
	variance = Statistics.var([1,2])
	epsilon=1e-6
	x.-=mean
	x*=(1/sqrt(variance+epsilon))
	x
end

nf = 3
resizes=(1,2,2, 2, 1) 
filters=(nf*8,nf*4,nf*2,nf,1)
k=5
kernel_size = (k,k)
total_resize = prod(resizes)
n_cells_x = n_cells+1
n_cells_y = n_cells+1
h = Int(n_cells_x / total_resize)
w = Int(n_cells_y / total_resize)
trainable_init = w*h*nf
dense_channels = nf

mₚ =  Chain(
	x->[1], # hack to not care about input 
	Dense(1, trainable_init; bias=false, ), # trainaible initialisation
	Dense(trainable_init, dense_channels*w*h,),
	x -> reshape(x,(w,h,dense_channels)),

	x->tanh.(x),
	Flux.unsqueeze(4),
	x->upsample_bilinear(x, (resizes[1],resizes[1])),
	x->Flux.normalise(x;dims=ndims(x)-1), 
	Conv(kernel_size, dense_channels => filters[1], pad=SamePad()),

	x->tanh.(x),
	x->upsample_bilinear(x, (resizes[2],resizes[2])),
	x->Flux.normalise(x;dims=ndims(x)-1), 
	Conv( kernel_size, filters[1] => filters[2], pad=SamePad()),

	x->tanh.(x),
	x->upsample_bilinear(x, (resizes[3],resizes[3])),
	x->Flux.normalise(x;dims=ndims(x)-1), 
	Conv( kernel_size, filters[2] => filters[3], pad=SamePad()),

	x->tanh.(x),
	x->upsample_bilinear(x, (resizes[4],resizes[4])),
	x->Flux.normalise(x;dims=ndims(x)-1), 
	Conv( kernel_size, filters[3] => filters[4], pad=SamePad()),

	x->tanh.(x),
	x->upsample_bilinear(x, (resizes[5],resizes[5])),
	x->Flux.normalise(x;dims=ndims(x)-1), 
	Conv( kernel_size, filters[4] => filters[5], pad=SamePad()),

	x->Flux.flatten(x)
)

p₀, m = Flux.destructure(mₚ)
mₓ(p,x) = m(p)(x) # close over a dummy input for x 

m(x) = p -> m(p,x)

# Defining the chain rules

function p_to_N(p)

	N = collect(Iterators.flatten(mₓ(p))) # convert the N x 1 array (the NN evaluated at p) to a vector 

end

function ChainRulesCore.rrule(::typeof(p_to_N),p)

	Nₘ, dNdp_vjp = Zygote.pullback(p->mₓ(p),p)
	N = collect(Iterators.flatten(Nₘ))

	function p_to_N_pullback(ds)

		dp = dNdp_vjp(Float32.(ds))[1]

		( NoTangent(),dp ) 
	end

	N, p_to_N_pullback

end

function N_to_L(N)

	sum( (NT.-N).^2 ) / length(NT)	

end


function ChainRulesCore.rrule(::typeof(N_to_L),N)

	function N_to_L_pullback(dL)

		dN = dL * ReverseDiff.gradient(N_to_L,N)

		( NoTangent(), dN )

	end

	N_to_L(N), N_to_L_pullback

end

function p_to_L(p)

	N   = p_to_N(p)
	L   = N_to_L(N)
	L

end

function ChainRulesCore.rrule(::typeof(p_to_L), p::AbstractVector)

	N,N_pullback   = rrule(p_to_N,p)
	L,L_pullback       = rrule(N_to_L,N)

	function p_to_L_pullback(dL)

		_, dN  = L_pullback(dL)
		_, dp     = N_pullback(dN)  

		( NoTangent(), dp)
	end

	return L, p_to_L_pullback

end

N₀ = p_to_N(p₀)
test_rrule(p_to_N,p₀;check_inferred=false,rtol=1e-2)
test_rrule(N_to_L,N₀;check_inferred=false,rtol=1e-2)
test_rrule(p_to_L,p₀;check_inferred=false,rtol=1e-2)

function fg!(F,G,p)
	jp,full_pullback = rrule(p_to_L,p)

	if G != nothing
	dj=1
	_,dp = full_pullback(dj)
	grads_vec = collect(Iterators.flatten(dp))
	copyto!(G,grads_vec)
	end
	loss_vec = collect(Iterators.flatten(jp))[1]
	return loss_vec
end

res = Optim.optimize(
		Optim.only_fg!(fg!), p₀, 
		LBFGS(),
		Optim.Options(
		iterations = 1000, store_trace=true, show_trace=true,
		allow_f_increases=true,
		)
		)

# Visualise result
pf = Optim.minimizer(res)
Nf = p_to_N(pf)
Nfₕ = FEFunction(Vbg,Nf)
writevtk(Ωbg,"CnnInterpolation",cellfields=["Nfₕ"=>Nfₕ,"NTₕ"=>NTₕ])

end #module 