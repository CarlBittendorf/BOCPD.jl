@testset "public API documentation surface" begin
    model = GaussianMeanModel()
    detector = BOCPDDetector(model)

    @test time_index(detector) == 0

    update!(detector, missing)

    @test observation_status(detector, 1) == :missing
    @test ismissingobservation(detector, 1)

    result = BOCPD.fit(model, Union{Missing,Float64}[0.0, missing]; history=FullHistory())

    @test time_index(result) == 2
    @test observation_status(result, 2) == :missing
    @test ismissingobservation(result, 2)
    @test length(changepoint_probabilities(result)) == 2
    @test length(changepoint_probabilities(result, FixedLag(0))) == 2
end
