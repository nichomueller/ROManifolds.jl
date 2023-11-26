module
using LinearAlgebra
using SparseArrays
using Serialization

import Gridap.Helpers.@check
import Gridap.Helpers.@notimplementedif
import Gridap.Arrays.LazyArray

const Float = Float64

export Float
export get_parent_dir,create_dir,correct_path,save,load,num_active_dirs
export vec_to_mat_idx,slow_idx,fast_idx,index_pairs,change_mode
export compress_array,recenter
export tpod,gram_schmidt!,orth_complement!,orth_projection

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("BasesConstruction.jl")
end # module
