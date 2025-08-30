The core ideas in operating systems — protection, concurrency, virtualization, resource allocation, and reliable storage — are widely used throughout computer science.

> [!important] Definition: operating system 
> An operating system is the layer of software that manages a computer’s resources system for its users and their applications.

Operating systems have three roles:
- Operating systems play referee —they manage shared resources between different applications running on the same physical machine. The OS isolate different applications from each other and protect itself and other applications.
- Operating systems play illusionist — they provide an abstraction physical hardware to simplify application design. OS provides two kind of illusion of a nearly infinite memory and that each program has the computer’s processors entirely to itself.
- Operating systems provide glue — a set of common services between applications.

> [!note] An Expanded View of an Operating System
> A portion of the operating system can also run as a library linked into each application. In turn, applications run in an execution context provided by the operating system. The application context is much more than a simple abstraction on top of hardware devices: applications execute in a virtual environment that is both more constrained (to prevent harm), more powerful (to mask hardware limitations), and more useful (via common services), than the underlying hardware.

![[Fig1.3.png]]