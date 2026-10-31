# Ocean temperature output metadata

The recovered ocean stores temperature in degrees Celsius. Its explicit lat/lon
adapter now labels that field `degC`, without converting or changing its values.
This correction belongs to the reference extraction itself, not only to the
constant-density candidate. The historical numerical capture remains unchanged.

The state-only density convenience uses the legacy linear EOS. Model-aware
output for explicitly selected dynamics is addressed separately in #24.
