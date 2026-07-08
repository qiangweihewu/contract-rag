from contract_rag.obs.tracer import NoopTracer, Tracer, TracerProtocol


def test_concrete_tracers_satisfy_protocol():
    assert isinstance(Tracer(), TracerProtocol)
    assert isinstance(NoopTracer(), TracerProtocol)


def test_non_tracer_is_not_a_tracer():
    assert not isinstance(object(), TracerProtocol)
