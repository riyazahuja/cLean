# Understanding Lean's Foreign Function Interface (FFI)

> **WARNING**
> 
> Note on FFI Interface Stability
> The current Foreign Function Interface (FFI) in Lean 4 was primarily designed for internal use within the Lean compiler and runtime. As such, it should be considered unstable. The interface may undergo significant changes, refinements, and extensions in future versions of Lean. Developers using the FFI should be prepared for potential breaking changes and should closely follow Lean's development and release notes for updates on the FFI system.

## Table of Contents

- [Understanding Lean's Foreign Function Interface (FFI)](#understanding-lean-s-foreign-function-interface--ffi-)
  * [1. Introduction](#1-introduction)
    + [What is FFI and Why Does it Matter?](#what-is-ffi-and-why-does-it-matter-)
    + [A Glimpse of Lean's FFI in Action](#a-glimpse-of-lean-s-ffi-in-action)
  * [2. FFI Scenarios](#2-ffi-scenarios)
    + [Mapping External Library to Lean](#mapping-external-library-to-lean)
    + [Exporting Lean Objects to External Libraries](#exporting-lean-objects-to-external-libraries)
  * [3. Boxed vs Unboxed Values in Lean's FFI](#3-boxed-vs-unboxed-values-in-lean-s-ffi)
    + [Unboxed Values](#unboxed-values)
    + [Boxed Values](#boxed-values)
    + [Boxed Scalars: A Special Case](#boxed-scalars--a-special-case)
  * [4. Scalar Values in Lean's FFI](#4-scalar-values-in-lean-s-ffi)
    + [Representation of Scalar Types](#representation-of-scalar-types)
    + [Boxing and Unboxing Functions](#boxing-and-unboxing-functions)
    + [Example: Working with Different Scalar Types](#example--working-with-different-scalar-types)
    + [Important Considerations](#important-considerations)
  * [5. Working with Lean Objects in C](#5-working-with-lean-objects-in-c)
    + [Types of Lean Objects](#types-of-lean-objects)
    + [Constructor Objects](#constructor-objects)
    + [Other Lean Object Types (Brief Overview)](#other-lean-object-types--brief-overview-)
  * [6. Working with C Objects in Lean](#6-working-with-c-objects-in-lean)
    + [Opaque Types](#opaque-types)
    + [Declaring Opaque Types](#declaring-opaque-types)
    + [Using Opaque Types](#using-opaque-types)
    + [Mapping C Structs to Lean Structures](#mapping-c-structs-to-lean-structures)
  * [7. Error Handling](#7-error-handling)
    + [Error Handling in C Code](#error-handling-in-c-code)
    + [Error Handling in Lean Code](#error-handling-in-lean-code)
    + [Error Handling Best Practices](#error-handling-best-practices)
  * [8. Parameter Ownership in Lean and FFI](#8-parameter-ownership-in-lean-and-ffi)
    + [Owned Parameters](#owned-parameters)
    + [Borrowed Parameters](#borrowed-parameters)
    + [FFI Considerations](#ffi-considerations)
  * [9 Building and Linking](#9-building-and-linking)
    + [Linking External Libraries](#linking-external-libraries)
    + [Lake Configuration](#lake-configuration)
  * [9 Best Practices](#9-best-practices)
  * [10. Further Resources](#10-further-resources)

## 1. Introduction

Lean, a powerful functional programming language and theorem prover, offers a robust Foreign Function Interface (FFI) that allows seamless interaction with C code. Whether you're a Lean enthusiast looking to leverage existing C libraries, optimize performance-critical code, or interface with low-level systems, understanding Lean's FFI is crucial.

In this article, we'll explore the intricacies of Lean's FFI system, focusing on how it represents different types of values, the concept of boxing and unboxing, and why these mechanisms are essential for efficient FFI operations.

### What is FFI and Why Does it Matter?

Foreign Function Interface (FFI) is a mechanism that enables code written in one programming language to call routines or use services written in another. For Lean, the primary use of FFI is to interact with C/C++ code.

The importance of FFI in Lean cannot be overstated:

1. It allows Lean programs to utilize the vast ecosystem of existing C and C++ libraries.
2. It provides a pathway for performance optimization in critical sections of code.
3. It enables direct interaction with system-level operations where C excels.
4. It facilitates the integration of Lean into existing systems or gradual migration of large codebases.

### A Glimpse of Lean's FFI in Action

Before we dive into the details, let's look at a simple example of how Lean's FFI works:

```lean
@[extern "c_add_ints"]
opaque addInts (x y : Int) : Int

#eval addInts 5 7  -- Should print 12
```

This Lean code declares an external function `addInts` that corresponds to a C function `c_add_ints`. Here's what the C function might look like:

```c
#include <lean/lean.h>

LEAN_EXPORT lean_obj_res c_add_ints(lean_obj_arg x, lean_obj_arg y) {
    long long result = lean_unbox_int64(x) + lean_unbox_int64(y);
    return lean_box_int64(result);
}
```

This simple example demonstrates the essence of FFI: calling C code from Lean and passing values between the two languages. FFI also allows C code to call Lean functions, which we'll explore later in this guide.

However, to use FFI effectively, we need to understand how Lean represents values when interfacing with C. This understanding forms the foundation for writing efficient and correct FFI code.

In the following sections, we'll explore the concepts of boxed and unboxed values, examine how Lean handles different types of values in FFI operations, and discuss the implications of these representations for FFI programming in Lean.

## 2. FFI Scenarios

Understanding the common scenarios in which FFI is used can help guide your implementation approach. When working with Lean's FFI, you'll typically encounter two primary scenarios:

* mapping external library functionality to Lean, and
* exporting Lean objects to external libraries.
  
Understanding these scenarios is crucial for effectively leveraging FFI in your Lean projects.

### Mapping External Library to Lean

In this scenario, you're primarily interested in using existing C libraries or functions within your Lean code. This is common when:

* You need to use a well-established C library that doesn't have a Lean equivalent.
* You're interfacing with system-level operations or hardware that require C code.
* You're optimizing performance-critical sections of your Lean code by implementing them in C.

When mapping external library values to Lean, you'll typically:

1. Declare external functions in Lean using the `@[extern]` attribute.
2. Create Lean types that correspond to C types, often using opaque types for complex C structures (`struct`, and `union` types)
3. Write C wrapper functions that convert between C and Lean representations.

Example: (FIXME: double-check this works)

```lean
-- Declaring an external C function in Lean
@[extern "c_read_file"]
opaque readFile (path : @& String) : IO ByteArray

-- Using the external function in Lean code
def main : IO Unit := do
  let contents ← readFile "example.txt"
  IO.println s!"File contents (size: {contents.size})"
```

```c
// C implementation of the external function
LEAN_EXPORT lean_obj_res c_read_file(b_lean_obj_arg path) {
    const char* c_path = lean_string_cstr(path);
    // Read file contents...
    // lean_object* lean_byte_array = ...; // Assume this is created from file contents
        return lean_io_result_mk_ok(lean_byte_array);
}
```

### Exporting Lean Objects to External Libraries

In this scenario, you're making Lean functionality available to C code. This is useful when:

* You're embedding Lean in a larger C/C++ application.
* You're gradually migrating a C codebase to Lean and need interoperability during the transition
* You're creating a Lean library that needs to be usable from C code.

When exporting Lean objects to external libraries (making Lean functionality available to C code), you'll typically:

1. Use the `@[export]` attribute to make Lean functions callable from C.
2. Design your Lean types with C interoperability in mind.
3. Provide C header files that declare the exported Lean functions.

Example:

```lean
structure Point where
  x : Float
  y : Float
deriving Repr

@[export lean_create_point]
def createPoint (x y : Float) : Point := { x := x, y := y }

@[export lean_point_distance]
def Point.distance (p1 p2 : Point) : Float :=
  let dx := p1.x - p2.x
  let dy := p1.y - p2.y
  Float.sqrt (dx * dx + dy * dy)
```

```c 
// C code using exported Lean functions
#include <lean/lean.h>

// Declarations of Lean-exported functions
extern lean_object* lean_create_point(double x, double y);
extern double lean_point_distance(lean_object* p1, lean_object* p2);

int main() {
    lean_object* p1 = lean_create_point(0.0, 0.0);
    lean_object* p2 = lean_create_point(3.0, 4.0);
    double distance = lean_point_distance(p1, p2);
    printf("Distance: %f\n", distance);
    lean_dec(p1);
    lean_dec(p2);
    return 0;
}
```

By understanding these two scenarios, you can choose the appropriate approach for your specific FFI needs, whether you're integrating external C libraries into your Lean code or making your Lean functionality available to C programs.


## 3. Boxed vs Unboxed Values in Lean's FFI

Understanding the distinction between boxed and unboxed values is crucial for effectively working with Lean's FFI. This knowledge applies to both scenarios: interfacing with external C libraries and exposing Lean functionality to C code.

### Unboxed Values

Unboxed values are stored directly as their primitive C type when they fit within a machine word (typically 32 or 64 bits, depending on the system architecture). These typically include:

* Small integers (like `UInt8`, `UInt16`, and `UInt32` and `Char` on 64-bit systems)
* Boolean values (`Bool`)

#### Mapping External Values to Lean

When calling C functions from Lean, unboxed values are passed directly, without any conversion overhead:

```lean
@[extern "c_add_uint32"]
opaque addUInt32 (x y : UInt32) : UInt32

#eval addUInt32 10 20  -- Should print 30
```

```c
LEAN_EXPORT uint32_t c_add_uint32(uint32_t x, uint32_t y) {
    // No boxing/unboxing needed for unboxed types
    return x + y;
}
```

#### Exporting Lean to External Libraries

When exporting Lean functions that work with unboxed values, they can be directly used in C code without boxing/unboxing:

```lean
@[export lean_add_uint32]
def addUInt32 (x y : UInt32) : UInt32 := x + y
```

```c
extern uint32_t lean_add_uint32(uint32_t x, uint32_t y);

int main() {
    uint32_t result = lean_add_uint32(10, 20);
    printf("Result: %u\n", result);
    return 0;
}
```

Notice how `UInt32` values are represented directly as `uint32_t` in C, without any need for boxing or unboxing.


### Boxed Values

Boxed values are Lean objects stored as pointers (`lean_object*`). They typically point to values that are larger than one machine word or need to be heap-allocated for other reasons (such as for uniformity in polymorphic contexts). These include:

* Complex types like non-trivial structures and inductive types
* Strings and arrays
* Larger scalar values that don't fit in a machine word (like `UInt64`, `UInt32` on 32-bit systems, `Float`, big `Int` and `Nat` numbers)
* Any value that needs to be treated polymorphically

#### Mapping External Values to Lean

When working with boxed values from C libraries, you'll need to use Lean's API functions to create and manipulate Lean objects:

```lean
@[extern "c_reverse_string"]
opaque reverseString (s : @& String) : IO String

#eval reverseString "Hello, Lean!"
```

```c
LEAN_EXPORT lean_obj_res c_reverse_string(lean_obj_arg s) {
    size_t len = lean_string_size(s);
    char* str = (char*)malloc(len + 1);
    strcpy(str, lean_string_cstr(s));

    for (size_t i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }

    lean_obj_res result = lean_mk_string(str);
    free(str);
    return lean_io_result_mk_ok(result);
}
```

In this example, we use Lean's string handling functions (`lean_string_size`, `lean_string_cstr`, `lean_mk_string`) to work with the boxed `String` object.

#### Exporting Lean to External Libraries

When exporting Lean functions that work with boxed values, C code needs to handle Lean objects and use Lean's API for manipulation:

```lean
structure Point where
  x : Float
  y : Float

@[export lean_create_point]
def createPoint (x y : Float) : Point := { x := x, y := y }

@[export lean_point_x]
def Point.x (p : Point) : Float := p.x
```

```c
extern lean_object* lean_create_point(double x, double y);
extern double lean_point_x(lean_object* p);

int main() {
    lean_object* point = lean_create_point(3.14, 2.71);
    double x = lean_point_x(point);
    printf("X coordinate: %f\n", x);
    lean_dec(point);
    return 0;
}
```

### Boxed Scalars: A Special Case

Lean uses a common optimization for small scalar values that need to be treated as objects. These "boxed scalars" are primitive values wrapped in a tagged `lean_object` pointer, but without additional heap allocation.

This approach is used when a value that could normally be unboxed needs to be treated as a Lean object, such as when stored in the field of a constructor object, for instance (note that Lean constructors support storing unboxed scalars too).

```c
lean_object* boxed_uint8 = lean_box(42);  // Boxing a UInt8
uint8_t unboxed = lean_unbox(boxed_uint8);  // Unboxing it later
```

By understanding these representations, you can write more efficient FFI code, properly manage memory, and ensure type safety when bridging between Lean and C in both directions.

In the next section, we'll dive deeper into how Lean handles different scalar types in FFI operations, including the specific boxing and unboxing functions for each type.

## 4. Scalar Values in Lean's FFI

Understanding how Lean handles scalar values (simple, single values like integers or floating-point numbers) in FFI operations is crucial for writing efficient and correct code. Let's explore how different scalar types are represented and the functions used for boxing and unboxing them.

### Representation of Scalar Types

In FFI operations, Lean represents scalar types as follows:

* `UInt8`, `UInt16`, and `Bool` are always passed unboxed in function calls.
* `UInt32` (and `Char` since it uses `UInt32`'s representation) is passed unboxed on 64-bit systems (where it fits in a machine word), but boxed on 32-bit systems (where it doesn't).
* `UInt64`, `USize`, and `Float` are always passed as boxed values.
* When these types need to be stored in Lean objects (like structures or inductive types), they are always boxed.

### Boxing and Unboxing Functions

Here's an overview of the functions used for boxing (converting a C value to a Lean object) and unboxing (converting a Lean object back to a C value) different scalar types:

| Lean Type | C Type    | Boxing Function              | Unboxing Function             |
|-----------|-----------|------------------------------|-------------------------------|
| `UInt8`   | `uint8_t` | `lean_box(size_t v)`         | `lean_unbox(lean_object* o)`  |
| `UInt16`  | `uint16_t`| `lean_box(size_t v)`         | `lean_unbox(lean_object* o)`  |
| `Bool`    | `uint8_t` | `lean_box(size_t v)`         | `lean_unbox(lean_object* o)`  |
| `UInt32`  | `uint32_t`| `lean_box_uint32(uint32_t v)`| `lean_unbox_uint32(lean_object* o)` |
| `Char`    | `uint32_t`| `lean_box_uint32(uint32_t v)`| `lean_unbox_uint32(lean_object* o)` |
| `UInt64`  | `uint64_t`| `lean_box_uint64(uint64_t v)`| `lean_unbox_uint64(lean_object* o)` |
| `USize`   | `size_t`  | `lean_box_usize(size_t v)`   | `lean_unbox_usize(lean_object* o)`  |
| `Float`   | `double`  | `lean_box_float(double v)`   | `lean_unbox_float(lean_object* o)`  |

### Example: Working with Different Scalar Types

Let's look at an example that demonstrates working with different scalar types in FFI:

```lean
@[extern "c_process_scalars"]
opaque processScalars (a : UInt8) (b : UInt32) (c : Float) : IO (UInt64 × Float)

#eval processScalars 5 1000 3.14
```

Here's the corresponding C function:

```c
LEAN_EXPORT lean_obj_res c_process_scalars(uint8_t a, uint32_t b, double c) {
    uint64_t result1 = (uint64_t)a * b;
    double result2 = c * 2;

    lean_object* tuple = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(tuple, 0, lean_box_uint64(result1));
    lean_ctor_set(tuple, 1, lean_box_float(result2));

    return lean_io_result_mk_ok(tuple);
}
```

In this example:

1. We pass `UInt8`, `UInt32`, and `Float` as unboxed values (when declared in an `@[extern ..]` function).
2. We perform calculations using these unboxed values.
3. We create a tuple (a Lean constructor object) to return tuple with two values.
4. We box the `UInt64` and `Float` results when storing them in the tuple.
5. We wrap the result in an `IO` monad using `lean_io_result_mk_ok`, as this is an `IO` operation in Lean.

You may ask why Lean stores `uint64_t` and `double` in the boxed fields and not unboxed in the scalars section. The reason is that `Prod` (`×`) is a polymorphic data structure and Lean does not monomorphize structures or inductives at this point

### Important Considerations

1. Platform Dependence: The behavior of some types (like `UInt32`) can vary between 32-bit and 64-bit systems. Always consider the target platform when writing FFI code.
2. Performance: Unboxed values are generally more efficient to work with in C code. When possible, design your FFI functions to work with unboxed scalar parameters and return values.
3. Type Checking: Lean's type system ensures correct usage on the Lean side, but in C code, you must be careful to use the correct boxing and unboxing functions for each type.
4. Memory Management: Boxed values are reference-counted. When creating or manipulating boxed values in C code, make sure to handle reference counting correctly to avoid memory leaks or use-after-free errors. Alternatively, you can use the `@&` attribute on individual parameters to pass values by reference. (We'll discuss this in more detail in the 'Parameter Ownership' section.)

By understanding these nuances of scalar representation and manipulation in Lean's FFI, you can write more efficient and correct code when interfacing between Lean and C.

## 5. Working with Lean Objects in C

When using Lean's FFI, you'll often need to interact with Lean objects in your C code. Lean objects are the C representation of Lean's data structures and values. Understanding the different types of Lean objects and how to handle them is crucial for effective FFI programming.

### Types of Lean Objects

In C, all Lean objects are represented as pointers of type `lean_object*`. However, there are several distinct types of Lean objects:

1. Constructor objects (`lean_ctor_object`)
2. Closures (`lean_closure_object`)
3. Arrays (`lean_array_object`)
4. Scalar arrays (`lean_sarray_object`)
5. Strings (`lean_string_object`)
6. Reference objects (`lean_ref_object`)
7. External objects (`lean_external_object`)
8. Thunks (`lean_thunk_object`)
9. Tasks (`lean_task_object`)

They all share a common header, the `lean_object` C structure that contains a tag, a field for reference counting, and other metadata.

Lean provides functions to check the type of an object:

```c
bool lean_is_scalar(lean_object* o);
bool lean_is_ctor(lean_object* o);
bool lean_is_closure(lean_object* o);
bool lean_is_array(lean_object* o);
bool lean_is_sarray(lean_object* o);
bool lean_is_string(lean_object* o);
bool lean_is_thunk(lean_object* o);
bool lean_is_task(lean_object* o);
```

Always check an object's type before performing operations on it to ensure type safety.

### Constructor Objects

Constructor objects represent Lean's structures and inductive values. They are the most common type of Lean object you'll work with in FFI code. For example, a Lean `Point` structure would be represented as a constructor object in C.

#### Layout of Constructor Objects

Constructor objects have a layout divided in two main sections (aside from the `lean_object` header):

1. Boxed fields (`lean_object*` pointers to other Lean objects)
2. Unboxed scalar fields of varying sizes

The layout for constructors does not follow the definition order found in the source code. Instead, it stores:
- **boxed fields**: these are all the `lean_object*` to boxed values in definition order
- **unboxed scalars**: this section contains all unboxed scalar ordered by their size from larger to smaller. Scalars with the same size following the definition order.

The layout of a constructor object in memory looks like this:

```
+----------------------------------+
| lean_object                      | |-- Header (tag, ref counting, ...)
+----------------------------------+
| (lean_object *)[0]               | \
| ...                              |  |-- n non-scalar fields
| (lean_object *)[n]               | /
+----------------------------------+
| size_t[n+1] + offset             | \
| ...                              |  |-- m USize fields
| size_t[n+m]                      |  |
| ---------------------------------| /
| uint64_t / double [n+m] + offset | \
| ...                              |  |-- UInt64 or Float
| uint64_t / double [n+m] + offset | /
| ---------------------------------|
| uint32_t[n+m] + offset           | \
| ...                              |  |-- UInt32 or Char
| uint32_t[n+m] + offset           | /
| ---------------------------------|
| uint16_t[n+m] + offset           | \
| ...                              |  |-- UInt16
| uint16_t[n+m] + offset           | /
| ---------------------------------|
| uint8_t[n+m] + offset            | \
| ...                              |  |-- UInt8
| uint8_t[n+m] + offset            | /
| ---------------------------------|
```

#### Working with Constructor Objects

Here's how you can work with constructor objects:

```c (FIXME: accessing scalar fields are not right)
// Check if it's a constructor object
if (lean_is_ctor(obj)) {
    // Get the number of fields
    unsigned num_fields = lean_ctor_num_objs(obj);
    
    // Get the constructor tag (useful for inductive types)
    unsigned tag = lean_obj_tag(obj);
    
    // Access a boxed field (0-based index)
    lean_object* field = lean_ctor_get(obj, 0);
    
    // Set a boxed field
    lean_ctor_set(obj, 0, new_value);
    
    // Access scalar fields
    // Note: Direct memory access like this should be used carefully
    uint8_t* scalar_ptr = lean_ctor_scalar_cptr(obj);
    uint32_t scalar_value = *((uint32_t*)(scalar_ptr + offset));
    
    // Set a scalar field
    // Caution: Ensure you're writing the correct type to the correct offset
    *((uint32_t*)(scalar_ptr + offset)) = new_scalar_value;
}
```

Example: Working with a Point Structure

```lean
structure Point where
  x : Float
  y : Float
deriving Repr

@[extern "c_distance"]
opaque distance (p1 p2 : Point) : IO Float
```

```c
#include <math.h>

LEAN_EXPORT lean_obj_res c_distance(b_lean_obj_arg p1, b_lean_obj_arg p2) {
    if (!lean_is_ctor(p1) || !lean_is_ctor(p2)) {
        return lean_io_result_mk_error(lean_mk_string("Expected Point objects"));
    }

    double x1 = lean_ctor_get_float(p1, 0);
    double y1 = lean_ctor_get_float(p1, 1);
    double x2 = lean_ctor_get_float(p2, 0);
    double y2 = lean_ctor_get_float(p2, 1);

    double dx = x2 - x1;
    double dy = y2 - y1;
    double distance = sqrt(dx*dx + dy*dy);

    return lean_io_result_mk_ok(lean_box_float(distance));
}
```

### Other Lean Object Types (Brief Overview)

While constructor objects are the most common, you may encounter other types of Lean objects in your FFI code. Here's a brief overview of each:

1. Closures: Represent Lean functions with captured variables. Use lean_closure_get_arity to get the arity of a closure.
2. Arrays: Contain a sequence of Lean objects. Use lean_array_size to get the size and lean_array_get to access elements.
3. Scalar Arrays: Similar to arrays but optimized for scalar values. Use lean_sarray_size and lean_sarray_get to work with them.
4. Strings: Represent Lean strings. Use lean_string_size to get the size and lean_string_cstr to get a C string representation.
5. Reference Objects: Represent mutable references in Lean. Use lean_ref_get and lean_ref_set to access and modify the referenced value.
6. External Objects: Used to wrap C data in Lean objects. Create them with lean_alloc_external and access the data with lean_get_external_data.
7. Thunks and Tasks: Represent delayed computations and potentially parallel tasks. These are more advanced and require careful handling of Lean's runtime.

When working with these object types, always use the appropriate Lean API functions (see [lean.h](https://github.com/leanprover/lean4/blob/master/src/include/lean/lean.h)) to ensure correct behavior and memory management.

**Note**: While this article focuses primarily on constructor objects, the other Lean object types mentioned here are also important for advanced FFI usage. We plan to cover these in more detail in a future article or in dedicated sections later in this guide. For now, be aware that these types exist and may require special handling in certain FFI scenarios.

## 6. Working with C Objects in Lean

This section covers strategies for representing and manipulating C objects within Lean.

### Opaque Types

Opaque types are used in Lean to represent C types whose internal structure is not directly accessible or relevant to Lean code. For example, a C `FILE*` pointer might be represented as an `opaque` type in Lean. They're particularly useful for C `struct`'s or `union`'s, or when you only need to pass the object between C functions without manipulating it in Lean.

### Declaring Opaque Types

To declare an opaque type in Lean:

```lean
opaque CFile : Type
```

This declares `CFile` as an opaque type that can be used to represent a C `FILE*` pointer.

Note that if you need an opaque type that implements the Inhabited type class (which is required for certain Lean operations), you can use the following workaround for now:

```lean
private opaque CXIndexPointed : NonemptyType
def CXIndex : Type := CXIndexPointed.type
instance : Nonempty CXIndex := CXIndexPointed.property
```

### Using Opaque Types

You can use opaque types in extern function declarations:

```lean
@[extern "c_fopen"]
opaque fopen (filename : @& String) (mode : @& String) : IO CFile

@[extern "c_fclose"]
opaque fclose (file : @& CFile) : IO Unit

@[extern "c_fprintf"]
opaque fprintf (file : @& CFile) (format : @& String) (value : @& String) : IO Unit
```

In this example, `CFile` is used to represent a C file pointer, allowing you to open, write to, and close files using C's standard I/O functions.

Note that `CFile` parameters are qualified as borrowed with `@&`. We'll discuss this in more detail in the 'Parameter Ownership' section.

### Mapping C Structs to Lean Structures

When dealing with C structs in Lean, there are several approaches, each with its own trade-offs in terms of performance, safety, and ease of use. We'll explore two main approaches: using C structs directly with accessor functions, and mapping C structs to Lean structures.

#### Using C Structs Directly with Accessor Functions

For optimal performance, especially when dealing with large or frequently accessed structures, it's best to work directly with C struct instances and use accessor functions. This approach avoids unnecessary allocation and copying between C and Lean.

Consider this C struct:

```c
typedef struct {
    int x;
    int y;
    const char* label;
} CPoint;

CPoint* create_cpoint(int x, int y, const char* label);
void free_cpoint(CPoint* p);
```

We can work with this in Lean using an opaque type and accessor functions:

```lean
opaque CPoint : Type

@[extern "create_cpoint"]
opaque CPoint.create (x y : Int32) (label : @& String) : IO CPoint

@[extern "free_cpoint"]
opaque CPoint.free (p : @& CPoint) : IO Unit

@[extern "c_point_get_x"]
opaque CPoint.getX (p : @& CPoint) : IO Int32

@[extern "c_point_set_x"]
opaque CPoint.setX (p : @& CPoint) (x : Int32) : IO Unit

-- Similar declarations for y and label
```

This approach is efficient because it operates directly on the C struct without copying data between C and Lean.

Here is how you can send and receive `CPoint` values to and from C:

```c
typedef struct {
    int x;
    int y;
    char* label;
} CPoint;

static CPoint* create_cpoint(int x, int y, const char* label) {
    CPoint* p = (CPoint*)malloc(sizeof(CPoint));
    if (p == NULL) return NULL;
    
    p->x = x;
    p->y = y;
    p->label = strdup(label);  // Make a copy of the label
    return p;
}

static void free_cpoint(CPoint* p) {
    if (p) {
        free(p->label);  // Free the label
        free(p);
    }
}

LEAN_EXPORT lean_obj_res c_create_cpoint(uint32_t x, uint32_t y, b_lean_obj_arg label) {
    const char* c_label = lean_string_cstr(label);
    CPoint* p = create_cpoint(x, y, c_label);
    if (p == NULL) {
        return lean_io_result_mk_error(lean_mk_string("Failed to create CPoint"));
    }
    // Note: The caller is responsible for freeing this CPoint later
    return lean_io_result_mk_ok(lean_box_usize((size_t)p));
}

LEAN_EXPORT lean_obj_res c_free_cpoint(b_lean_obj_arg p_obj) {
    CPoint* p = (CPoint*)lean_unbox_usize(p_obj);
    free_cpoint(p);
    return lean_io_result_mk_ok(lean_box(0));  // Return IO Unit
}

LEAN_EXPORT lean_obj_res c_point_get_x(b_lean_obj_arg p_obj) {
    CPoint* p = (CPoint*)lean_unbox_usize(p_obj);
    return lean_io_result_mk_ok(lean_box_uint32(p->x));
}

LEAN_EXPORT lean_obj_res c_point_set_x(b_lean_obj_arg p_obj, uint32_t x) {
    CPoint* p = (CPoint*)lean_unbox_usize(p_obj);
    p->x = x;
    return lean_io_result_mk_ok(lean_box(0));  // Return IO Unit
}
```

Note how `lean_box_usize` and `lean_unbox_usize` is used to box and unbox a pointer to `CPoint`. This is how an opaque is represented in Lean.

#### Mapping C Structs to Lean Structures

In some cases, you might want to create a Lean-managed copy of a C struct. This can be useful when:

1. Working with small, stack-allocated C structs that you want to persist in Lean beyond the C function call.
2. You need to work with the data extensively in Lean and want to leverage Lean's type system and memory management.
3. The C struct is simple enough that the overhead of copying is negligible compared to the benefits of working with a native Lean structure

Here's an example of mapping a simple C struct to a Lean structure:

```c
typedef struct {
    int x;
    int y;
} CPoint;

CPoint get_point();
```

In Lean:

```lean
structure Point where
  x : Int32
  y : Int32
deriving Repr

@[extern "get_point"]
opaque getPoint : IO Point

def main : IO Unit := do
  let point ← getPoint
  IO.println s!"Point: {point}"
```

The C implementation might look like:

```c
LEAN_EXPORT lean_obj_res get_point() {
    CPoint c_point = get_point();  // Get the C struct
    lean_object* lean_point = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(lean_point, 0, lean_box_int32(c_point.x));
    lean_ctor_set(lean_point, 1, lean_box_int32(c_point.y));
    return lean_io_result_mk_ok(lean_point);
}
```

This approach involves copying data from the C struct to a Lean structure, which is then managed by Lean's memory system. One example where this approach may be used is when a C library allocates a struct in the stack and you want to send that value back to Lean.

## 7. Error Handling

Proper error handling is crucial when working with Lean's Foreign Function Interface (FFI). It ensures robustness and helps in debugging. In FFI, errors can occur in both C and Lean code, so it's important to handle them appropriately on both sides

### Error Handling in C Code

When implementing C functions for Lean FFI, use `lean_io_result_mk_ok` to return a successful result, and `lean_io_result_mk_error` to return an error for IO operations.

```c
LEAN_EXPORT lean_obj_res c_fallible_io_operation(...) {
    if (/* error condition */) {
        // Note: lean_mk_string creates a new string object that Lean will manage
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Operation failed")));
    }
    // ... perform operation ...
    return lean_io_result_mk_ok(lean_box(0)); // equivalent to `return ()`
}
```

For pure functions returning `Option`, you need to return a constructor with the correct tag:

```lean
inductive Option (α : Type u) where
  | none : Option α                  -- tag == 0
  | some (val : α) : Option α        -- tag == 1
```

For pure functions that might fail, returning an `Option` (or `Except`) is often more appropriate than using `IO`. Here's how to return an `Option` in C:

```c
LEAN_EXPORT lean_obj_res c_fallible_operation(...) {
    if (/* error condition */) {
        return lean_alloc_ctor(0, 0, 0); // Option.none
    }
    // ... perform operation ...
    lean_object* some = lean_alloc_ctor(1, 1, 0);
    lean_ctor_set(some, 0, lean_box(0));
    return some; // Option.some ()
}
```

### Error Handling in Lean Code

When working with FFI functions in Lean, use the IO monad for operations that might fail:

```lean
@[extern "c_fallible_io_operation"]
opaque fallibleIOOperation : Unit -> IO Unit

def safeOperation : IO Unit := do
  try
    fallibleIOOperation ()
  catch e =>
    IO.println s!"Error occurred: {e}"
    -- Note: You might want to rethrow the error or return a specific error value here
```

For pure functions returning `Option`:

```lean
@[extern "c_fallible_operation"]
opaque fallibleOperation : Unit -> Option Unit

def processResult : () -> IO Unit := do
  match fallibleOperation () with
  | none => IO.println "Operation failed"
  | some _ => IO.println "Operation succeeded"
```

### Error Handling Best Practices

1. Be specific in error messages to aid debugging.
2. Use Lean's IO monad for operations with side effects or that may fail.
3. Consider using Option or Except for pure functions that may fail.
4. Always handle potential errors when calling FFI functions in Lean code.
5. Use try-catch blocks in Lean to manage exceptions from IO FFI calls.
6. For complex FFI operations, consider creating custom error types in Lean.
7. Ensure consistency in error handling between C and Lean code. Errors should be propagated and handled similarly regardless of where they originate.

## 8. Parameter Ownership in Lean and FFI

Understanding parameter ownership is crucial for proper memory management in Lean, especially when working with FFI. In FFI scenarios, mismanaging ownership can lead to memory leaks or use-after-free errors, which can be particularly hard to debug across language boundaries. In Lean, including in FFI contexts, there are two types of parameter ownership: **owned** (default) and **borrowed**.

### Owned Parameters

By default, all parameters in Lean are owned. Ownership in this context relates to reference counting and memory management. This means:

1. Before passing a value as an **owned** parameter, Lean **increments** the value's reference count.
2. The function receiving the parameter is responsible for:
   1. **Incrementing** the reference count when passing it to other functions
   2. **Decrementing** the reference count when finished using the value.

### Borrowed Parameters

Parameters can be declared as **borrowed** by prefixing them with `@&`. For example: `def someFunction (x : @& MyType) : IO Unit`. For borrowed parameters:

1. Lean does not increment the reference count when passing the value.
2. The function implementation does not need to manage reference counting for this parameter.
3. The function is expected to only use the value as a reference, without taking ownership.

### FFI Considerations

In FFI contexts, values representing foreign objects (like C structs or pointers) often don't require Lean's reference counting management. These can be efficiently declared as borrowed parameters to avoid unnecessary overhead. For example:

```lean
@[extern "c_point_get_x"]
opaque CPoint.getX (p : @& CPoint) : IO Int32
```

Here, the `p` parameter is borrowed, simply carrying a boxed pointer to a C object without involving Lean's reference counting mechanism.

Note: When using borrowed parameters (`@&`) in Lean, ensure that the corresponding C function treats these parameters as borrowed (using `b_lean_obj_arg`). Consistency in borrowing across the language boundary is crucial for correct memory management.

In general, use borrowed parameters (@&) when you only need to read from an object without storing it or passing ownership. Use owned parameters when you need to store the object or pass ownership to another function. When in doubt, using owned parameters is safer but may incur a small performance cost due to reference counting.

## 9 Building and Linking

Lake, Lean's built-in build system and package manager, simplifies the process of building and linking Lean projects that use FFI. FFI projects require special build configurations to compile C code and link with external libraries alongside Lean code. This section covers how to configure your project for both mapping external library values to Lean and exporting Lean objects to external libraries.

### Linking External Libraries

When building a Lean project that uses external C libraries, you need to configure Lake to:

1. Compile the wrapper `.c` files created to bridge between Lean and the external library, and
2. Link the external library

These steps ensure that your Lean code can call C functions and that all necessary external code is included in the final executable.

### Lake Configuration

There are myriad ways to use Lake to build and link external libraries. The approach presented here is a simple configuration that works well for many common scenarios. The specifics will depend on what is required by the external library and the kind of automation you want to put in place to deal with platform specific nuances, download sources, build, etc.

The configuration suggested here is a simple Lake configuration that works well and focus only on building and linking:

```lean
import Lake
open Lake DSL

package "simple" where
  moreLinkArgs := #[
    "-llibname",                   
    "-L/path/to/your/external/lib"
  ]

/--
Given a Lean module named `M.lean`, build a C shim named `M.shim.c`
-/
@[inline] private def buildCO (mod : Module) (shouldExport : Bool) : FetchM (BuildJob FilePath) := do
  let cFile := mod.srcPath "shim.c"
  let irCFile := mod.irPath "shim.c"
  let cJob ← -- get or create shim.c file (we no shim.c is found, create an empty one to make lake happy)
    if (← cFile.pathExists) then
      proc { cmd := "cp", args := #[cFile.toString, irCFile.toString]}
      inputTextFile irCFile
    else
      logVerbose s!"creating empty shim.c file at {irCFile}"
      let _<-  proc { cmd := "touch", args := #[irCFile.toString] }
      inputTextFile irCFile

  let oFile := mod.irPath s!"shim.c.o.{if shouldExport then "export" else "noexport"}"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString] ++ mod.weakLeancArgs
  let cc := (← IO.getEnv "CC").getD "clang"
  let leancArgs := if shouldExport then mod.leancArgs.push "-DLEAN_EXPORTING" else mod.leancArgs
  buildO oFile cJob weakArgs leancArgs cc

module_facet shim.c.o.export mod : FilePath := buildCO mod true
module_facet shim.c.o.noexport mod : FilePath :=  buildCO mod false

lean_lib «Simple» where
  nativeFacets := (#[Module.oExportFacet, if · then `shim.c.o.export else `shim.c.o.noexport])

@[default_target]
lean_exe "simple" where
  root := `Main
```

This configuration sets up Lake to automatically build C shim files alongside your Lean files, link against the specified external library, and include the necessary compiler and linker flags. It provides a foundation that you can customize based on your project's specific needs.

This simple configuration will:
1. Automatically build any `.shim.c` C file you add next to a `.lean` file (e.g., `Simple.shim`.c for `Simple.lean`)
2. Automatically build any `.shim.c` C file you add next a `.lean` (e.g. `Simple.shim.c` for `Simple.lean`)
3. Link against the external library

Note that this is just a starting point to get you building and linking your FFI project. You may need to adjust it based on your specific project requirements.

To use this configuration:

1. Modify the `moreLinkArgs` in the package configuration to point to your external library.
2. Or, make sure that your compiler can find the library specified in `moreLinkArgs` (e.g. `DYLD_LIBRARY_PATH`, `LD_LIBRARY_PATH`, etc depending on your platform).
3. Create `ModuleName.shim.c` files alongside your `ModuleName.lean` files for modules that require FFI.
4. Run `lake build` to compile your project.
5. Be aware that you may need to adjust this configuration for different platforms or build environments. Always test your build on all target platforms.

## 9 Best Practices

When working with Lean's Foreign Function Interface (FFI), following these best practices will help you write efficient, safe, and maintainable code. FFI introduces additional complexity due to the interaction between different programming languages, making adherence to best practices crucial for preventing subtle bugs and performance issues.

1. Type Safety
    * Always check object types before operating on them in C code.
    * Use Lean's type system to your advantage when designing FFI interfaces: use typed, opaque types to deal with pointers. For example:
    ```lean
    opaque CFileHandle : Type
    @[extern "c_open_file"]
    opaque openFile (path : String) : IO CFileHandle
    ```
2. Reference counting is crucial for proper memory management in Lean. Mismanaging reference counts can lead to memory leaks or use-after-free errors. When working with FFI:
    * For owned parameters:
        - Increment the reference count when storing an object for later use.
        - Decrement the reference count when you're done with an object.
    * For borrowed parameters (marked with `@&` in Lean):
        - Use `b_lean_obj_arg` in C code instead of `lean_obj_arg`.
        - Avoid reference count operations on borrowed objects.
3. Object Initialization
    * When creating new Lean objects in C code, ensure all fields are properly initialized before the object can be accessed by Lean code. For constructor objects, use `lean_ctor_set` to initialize all fields.
4. Error Handling
    * Handle errors gracefully in C code.
    * Return errors as Lean `IO` error results using `lean_io_result_mk_error`.
    * Ensure consistency in error handling between C and Lean code. Errors should be propagated and handled similarly regardless of where they originate.
5. Memory Management
    * Be vigilant about memory leaks, especially when working with C-allocated resources. Always free C-allocated memory when it's no longer needed, and use Lean's reference counting system correctly for Lean objects.
6. FFI Design
    * Keep FFI interfaces as simple as possible to minimize the risk of errors.
    * Use opaque types in Lean and `@[extern]` accessors for complex C structures
    * Document your FFI interfaces thoroughly, including any assumptions about memory ownership, threading, or error handling. This is especially important because type information alone may not convey all necessary usage details.

By following these guidelines and understanding how to work with Lean objects in C, you can write efficient and safe FFI code, bridging the gap between Lean's powerful type system and C's low-level control.


## 10. Further Resources

To deepen your understanding of Lean's FFI and expand your skills, consider exploring these additional resources:

1. Lean Documentation: The official Lean documentation provides in-depth information about the language and its features, including FFI

2. Lean 4 Manual

3. Lean Source Code: Examining the Lean source code, particularly the parts dealing with FFI, can provide valuable insights.

4. Lean 4 GitHub Repository

5. Lean Zulip Chat: The Lean community is active and supportive. Engage with other Lean users to share experiences and get help.

6. C Programming Resources: Since Lean's FFI interacts primarily with C, refreshing your C programming skills can be beneficial.
    * "The C Programming Language" by Brian Kernighan and Dennis Ritchie
    * C Reference on cppreference.com

7. "The C Programming Language" by Brian Kernighan and Dennis Ritchie C Reference on cppreference.com

Remember, FFI is a complex topic, and mastery comes with practice. Start with small projects, gradually increasing complexity as you become more comfortable with the concepts and techniques. By leveraging Lean's FFI effectively, you can create powerful, efficient, and interoperable software that combines the best of Lean's theorem proving capabilities and type safety with C's performance and vast ecosystem of libraries.
