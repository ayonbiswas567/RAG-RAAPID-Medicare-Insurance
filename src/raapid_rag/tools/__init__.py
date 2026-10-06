"""Mock external tools wired into the model's reasoning loop via function calling.

Each tool is a plain function returning a JSON-serialisable dict.  Errors are returned as
``{"error": "..."}`` rather than raised, so the model can see what went wrong and report
missing information instead of the request failing.  The registry adds timeouts and
argument validation on top.
"""
