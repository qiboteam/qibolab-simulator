# Direct platform loading and Marimo shutdown

The notebook now calls `qibocal.create_calibration_platform("split_transmons")`
directly. There is no local platform wrapper or signal-handler manipulation.
The example's `calibration.json` provides the initial metadata formerly
constructed by the deleted helper.

Direct acquisition, fitting, reporting, and updates work. However, headless
Marimo HTML export writes its output and then hangs at shutdown with the current
Qibolab 0.2.16 dependency and Qibocal commit
`cdd2133f0aa44db150bd935f22a87e0da40898c5`. The export process remained alive
after 120 seconds and had to be stopped. The same export previously terminated
with the helper's signal-handler restoration.

Reproduce from the repository root:

```sh
uv run --locked --group notebooks marimo export html notebooks/calibration.py \
    -o /tmp/calibration.html -- --protocol rabi_amplitude
```

Qibolab's `Platform.__post_init__` installs handlers for `SIGTERM` and `SIGINT`.
Its `termination_handler` disconnects and raises `RuntimeError`, replacing the
notebook runner's existing shutdown behavior. The implementation is in
`qibolab/_core/platform/platform.py`, lines 75-78 and 171-175 in Qibolab 0.2.16.
This is the apparent source of the shutdown regression, rather than a failure
in protocol execution or failure to disconnect the instruments.

The existing docs task uses this export command, so its completion is affected.
An upstream lifecycle fix should make process-wide signal handling opt-in or
leave it to the application embedding the platform. No notebook-side workaround
has been reintroduced.
