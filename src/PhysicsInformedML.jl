module PhysicsInformedML

import SVR
import NMFk
import Mads
import Printf
import Suppressor

if Base.source_path() !== nothing
	const dir = first(splitdir(first(splitdir(Base.source_path()))))
end

include("PhysicsInformedML_Model.jl")
include("PhysicsInformedML_Mads.jl")

end