"""Environment-independent primitives: durable artifacts and step metering.

Nothing in this package imports a simulator, a deep-learning framework or
CUDA. That is deliberate: it keeps the protocol auditable and lets the test
suite run in milliseconds.
"""
