# Direct built-in protocols: gaps in qiboteam/qibocal#1716

The requested simplification is blocked by the direct execution interface, not
by missing binding syntax. Rather than introduce another compatibility layer,
leave `examples/calibration.py` and `notebooks/calibration.py` unchanged until
Qibocal supports direct execution of its built-in `Protocol` instances.

This assessment refers to
[qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716) at commit
`8f2f434cae99193dc1cbe6a6d2e40a157f656944`. This repository currently pins the
older commit `3b15fde3cc229838b093a402cc5e45e3e35e92ed` in `pyproject.toml`.

## Binding is already implemented

[`Protocol.__call__`](https://github.com/qiboteam/qibocal/blob/8f2f434cae99193dc1cbe6a6d2e40a157f656944/src/qibocal/auto/operation.py#L325-L349)
already has the signature proposed in
[qiboteam/qibocal#1456](https://github.com/qiboteam/qibocal/issues/1456):

```python
protocol(pars=None, fit=None, report=None, **kwargs)
```

It returns a `BoundProtocol` retaining the original protocol instance, without
creating an executor. No local replacement `bind()` is needed.

However, keyword construction currently uses `parameters_type(**kwargs)`.
Built-in parameter dataclasses inherit from the non-dataclass `Parameters`:
`nshots` and `relaxation_time` are annotations on the base class, not constructor
fields. For example, `rabi_amplitude(..., nshots=4096, relaxation_time=0)` raises
`TypeError` for an unexpected constructor keyword.
[`Parameters.load`](https://github.com/qiboteam/qibocal/blob/8f2f434cae99193dc1cbe6a6d2e40a157f656944/src/qibocal/auto/operation.py#L62-L99)
currently handles these fields separately. Passing its output as `pars` is a
small existing workaround, but upstream keyword binding should also support
the built-in execution parameters without breaking plain parameter dataclasses.

Protocol instances also do not supply defaults for required sweep fields.
Removing the current `parameters()` sweep presets therefore means exposing
explicit experiment inputs in the notebook, not relying on binding to invent
them. Optional protocol fields should retain their upstream defaults.

## Direct execution does not support the built-ins

The phase calls in
[`Executor`](https://github.com/qiboteam/qibocal/blob/8f2f434cae99193dc1cbe6a6d2e40a157f656944/src/qibocal/auto/execute.py#L127-L196)
do not match the callbacks of
[`protocols.rabi_amplitude`](https://github.com/qiboteam/qibocal/blob/8f2f434cae99193dc1cbe6a6d2e40a157f656944/src/qibocal/protocols/rabi/amplitude.py),
which is a representative built-in used by this notebook:

| Phase | Direct executor calls | Built-in callback expects |
| --- | --- | --- |
| Acquisition | `acquisition(parameters)` | `acquisition(parameters, platform, targets)` |
| Fit | `fit(data, fitpars)` | `fit(data)` |
| Report | `report(data, results, reportpars)` | `report(data, target, fit=None)` returning figures and a table |
| Update | `update(results, platform)` | `update(results, platform, target)` |

Consequently, binding an existing parameter object succeeds, but
`executor.acquire(bound)` fails before acquisition because `platform` and
`targets` are missing. Setting `Executor(platform, targets=[0, 1])` does not
fix this: direct acquisition uses neither value. Fit passes an extra argument;
report assigns results to the target argument and discards the callback's
return value; update lacks the target argument.

The latest PR adapts legacy acquisition and fit callbacks inside
[`Task.run`](https://github.com/qiboteam/qibocal/blob/8f2f434cae99193dc1cbe6a6d2e40a157f656944/src/qibocal/auto/task.py#L166-L195).
Those adapters create a new `Protocol` and are not available through the direct
bound-protocol API. Routing the notebook through tasks, actions, output folders,
or another locally constructed protocol would defeat the requested simplification.

## Upstream ingredients needed

Provide a supported direct execution path for the actual built-in instances,
with an explicit contract for platform and target selection. Migration of
built-in callbacks or an upstream compatibility mechanism could both work;
the notebook should not have to implement that mechanism itself.

The contract needs to cover all four phases, not acquisition alone. In
particular, reporting must expose figures and tables to the caller, and updates
must apply to the selected targets on the executor's actual platform. The
current report API returns `None`, while the notebook must render the returned
Plotly figures and HTML tables. Report parameters should remain parameters,
not require a mutable output list as a side channel.

Define platform lifecycle ownership for direct acquisition, or explicitly
document caller-managed `connect()` / `disconnect()` with exception-safe
cleanup. The current notebook adapter owns that lifecycle; removing it must
not leave instruments connected after failures. Preserve independent phase
execution and opt-in updates to the in-memory session, without filesystem
registration or runcard writes.

The fit-only binomial error floor and missing-target fit-result check are
not protocol-binding responsibilities. When simplifying this notebook, they
can remain explicit small data-processing steps or be removed deliberately;
they should not justify wrapping every phase in a new `Protocol`.

Useful upstream regression coverage would bind `protocols.rabi_amplitude`
both from keywords (including `nshots` and `relaxation_time`) and from `pars`,
then acquire, fit, retrieve reports, and update only selected targets using a
mock calibration platform. It should assert that binding preserves protocol
identity and needs no executor, and exercise independent phases as well as
`Executor(..., update=False)`. The new-style custom spectroscopy demonstration
does not exercise the built-in callback signatures.

## Follow-through in this repository

Once those contracts exist, update the Qibocal dependency pin and simplify the
notebook around direct built-in protocol calls and a separately created
`Executor`. Keep platform construction separate, use explicit acquisition
parameters instead of a protocol-specific option factory, and remove
`UpdatePlatform` and the acquisition/fit/report/update closures from the wrapper.

The dependency pin also matters for updates: the pinned revision exposes
`executor.update(results)` delegating to `platform.update(results)`, whereas
the inspected PR head exposes `executor.update(results, bound)` invoking the
protocol callback. The existing notebook intentionally follows the old API;
changing just its binding function or just its pin is insufficient.
