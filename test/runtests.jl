using BOCPD
using Test
using Aqua

@testset "BOCPD.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BOCPD)
    end
    # Write your tests here.
end
