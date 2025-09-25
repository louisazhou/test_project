# Coding Standards and Best Practices

## String Formatting
**Rule 1: Meaningful f-strings only**
- If an f-string is used, it must contain at least one variable substitution
- Don't use f-string for static strings - remove the `f` prefix if no variables are involved
- ❌ Bad: `f"This is a static string"`
- ✅ Good: `"This is a static string"` or `f"Value is {variable}"`

## Function Definitions
**Rule 2: Avoid disposable one-liner functions**
- Lambda functions are acceptable for short, reusable logic
- Avoid creating one-liner disposable functions that are only used once
- Prefer inline logic or meaningful function names for reusable code
- ❌ Bad: `def temp_func(x): return x + 1` (used once)
- ✅ Good: `lambda x: x + 1` (for simple transforms) or proper function names

## Lambda Function Scope
**Rule 3: Ensure proper lambda scope**
- Lambda functions must have access to all variables they reference
- Avoid closure issues by ensuring variables are properly bound
- Pass required parameters explicitly rather than relying on closure
- ❌ Bad: `lambda x: _match_parent(parent_label, x, parent_index)` (parent_label not in scope)
- ✅ Good: `lambda x, parent_label=parent_label: _match_parent(parent_label, x, parent_index)`

## Dictionary Creation
**Rule 4: Use dictionary literals**
- Use `{}` literal syntax instead of `dict()` constructor when possible
- This is more readable and slightly more efficient
- ❌ Bad: `dict(key1=value1, key2=value2)`
- ✅ Good: `{"key1": value1, "key2": value2}`

## Variable Scope
**Rule 5: Proper variable scope in functions**
- Ensure variables inside functions have the correct scope
- **Nested functions must NOT use closure variables** - pass all dependencies as explicit parameters
- Make dependencies explicit through function parameters, especially for nested functions
- ❌ Bad: `def inner(): return outer_var` (uses closure)
- ✅ Good: `def inner(outer_var): return outer_var` (explicit parameter)
- ❌ Bad: Nested functions relying on outer scope variables without parameters
- ✅ Good: Pass all required values as explicit parameters to nested functions

## Variable Naming
**Rule 6: Avoid ambiguous single-letter variables**
- Do not use "I" as a variable name (too similar to "l" or "1", can cause serious bugs)
- Use descriptive names like "total_impact" instead of "I"
- Indexed variants are acceptable: "I_0", "I_1", "I_2"
- ❌ Bad: `I = some_calculation()`
- ✅ Good: `total_impact = some_calculation()` or `I_final = some_calculation()`

## Implementation Notes
These rules help maintain:
- Code clarity and readability
- Predictable variable scope and lifetime
- Performance optimization (avoiding unnecessary f-string processing)
- Maintainability and debugging ease
