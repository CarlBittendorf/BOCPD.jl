@testset "BOCPDResult display" begin
    data = [0.0, 1.0, 0.0, 3.0]
    result = BOCPD.fit(GaussianMeanModel(), data; hazard=GeometricHazard(0.2), history=FixedLagHistory(2))

    compact = sprint(show, result)
    @test occursin("BOCPDResult", compact)
    @test occursin("time", compact)
    @test !occursin("runlength_history", compact)
    @test !occursin("discarded_mass", compact)
    @test !occursin("nothing", compact)
    @test length(compact) < 220

    empty = BOCPD.fit(GaussianMeanModel(), Float64[]; hazard=ConstantHazard(10))
    @test occursin("BOCPDResult", sprint(show, empty))
    @test occursin("0 time points", sprint(show, empty))

    plain = sprint(
        show,
        MIME("text/plain"),
        result;
        context = IOContext(stdout, :limit => true, :compact => false, :displaysize => (24, 80)),
    )
    @test startswith(plain, "BOCPDResult")
    @test occursin("time points", plain)
    @test occursin("retention", plain)
    @test occursin("maximum delay", plain)
    @test !occursin("missing observations", plain)
    @test !occursin("runlength_history", plain)
    @test !occursin("discarded_mass", plain)

    missing_result = BOCPD.fit(
        GaussianMeanModel(),
        Union{Missing,Float64}[0.0, missing, 1.0];
        hazard=ConstantHazard(10),
        history=FixedLagHistory(2),
    )
    plain_missing = sprint(
        show,
        MIME("text/plain"),
        missing_result;
        context = IOContext(stdout, :limit => true, :compact => false, :displaysize => (24, 80)),
    )
    @test occursin("missing observations", plain_missing)
    @test occursin("1", plain_missing)
    @test !occursin("runlength_history", plain_missing)
    @test !occursin("raw", plain_missing)

    narrow = sprint(
        show,
        MIME("text/plain"),
        result;
        context = IOContext(stdout, :limit => true, :compact => false, :displaysize => (24, 40)),
    )
    @test startswith(narrow, "BOCPDResult")
    @test occursin("time points", narrow)
    @test length(narrow) < 500

    full_history = BOCPD.fit(GaussianMeanModel(), data; hazard=ConstantHazard(10), history=FullHistory())
    full_plain = sprint(
        show,
        MIME("text/plain"),
        full_history;
        context = IOContext(stdout, :limit => true, :compact => false, :displaysize => (24, 80)),
    )
    @test occursin("FullHistory", full_plain)
    @test occursin("maximum delay", full_plain)
end
