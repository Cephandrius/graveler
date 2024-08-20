# Dear Austin, Hi! It's me, Christopher.
I made two different attempts to run the graveler simulation faster. One in python and one using C and OpenCL. With the python code, I got rid of the lists you were using. We only really need to keep track of successes, not the individual results. I also used random.random() for the randomness. It is significantly faster than random.choice().

# OpenCL
OpenCL is a library for running programs in parallel, meaning that you can run many instances of the same program at the same time.  Graphics cards are designed to run in parallel, so that what I used to beat your time. I'm guessing that you'll want to run the code on your computer, so I'll guide you through it. 

[This](https://github.com/KhronosGroup/OpenCL-Guide) is a github to get you introduced to OpenCL. It has a couple of pages for getting OpenCL installed.

If you got it installed okay, then you can download my code from github. Navigate to src\opencl\ and run
'''
cmake -S . -B build\
cmake --build .\build
.\build\graveler 1000000000
'''
There might be some things that go wrong and I would love to help you out. Reach out to me a paulchristopherschuckjr@gmail.com .
