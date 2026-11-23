# Security Policy

## Supported Versions

Only the latest release of `sonnix` on PyPI receives security fixes. Please
upgrade to the latest version before reporting an issue.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report them privately with
[GitHub Security Advisories](https://github.com/durandtibo/sonnix/security/advisories/new)
instead.

You should receive an acknowledgment within 5 business days. After that, we
will give you an initial assessment and a fix timeline within 10 business days.

Please include as much of the following information as possible to help us
understand and reproduce the issue:

- The type of issue (for example denial of service, unsafe deserialization or
  code execution).
- The paths of the source files related to the issue.
- The location of the affected code (tag, branch, commit or URL).
- Any special configuration needed to reproduce the issue.
- Step-by-step instructions to reproduce the issue.
- Proof-of-concept or exploit code, if possible.
- The impact of the issue, including how an attacker might exploit it.

## Update Policy

Security fixes are released as soon as possible. They are marked as such in the
[changelog](CHANGELOG.md) and announced in the GitHub releases.

## Security Considerations

`sonnix` is a library of PyTorch modules, losses and functional operations. It
does not load files, open network connections or execute code from strings.
Keep the following in mind when you use it:

- **Untrusted checkpoints**: `sonnix` modules are `torch.nn.Module` objects.
  Loading a checkpoint or a pickled module with `torch.load` can execute
  arbitrary code. Only load files from trusted sources, and prefer
  `torch.load(..., weights_only=True)` or safetensors for state dicts.
- **Resource exhaustion**: very large inputs, or arguments that create large
  tensors (for example large dimensions or sizes), can consume a lot of memory
  and compute. Validate the sizes of the data that come from untrusted sources.
- **Numerical robustness**: inputs with `nan` or `inf` values propagate through
  the modules and losses. Do not rely on them to sanitize untrusted data.
- **Dependencies**: `sonnix` relies on `torch` and `coola`. Keep them up to date
  to get their security fixes.
