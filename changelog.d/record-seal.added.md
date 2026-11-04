A reference collector, `python -m osprey.agent_record.collector`, seals the
first head it receives for a day with an Ed25519 key it holds off the host (a
test key it makes on first run) and refuses a second, different one;
`python -m osprey.agent_record.verify` checks a copy of the archive against
those seals and rejects a day with an edited, deleted or inserted entry, or a
dropped day.
