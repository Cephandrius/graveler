#Dear Austin, Hi! It's me, Christopher.
I made two different attempts to run the graveler simulation faster. One in python and one using C and OpenCL. With the python code, I got rid of the lists you were using. We only really need to keep track of successes, not the individual die rolls. I also used random.random() for the randomness. It is significantly faster than random.choice().

#OpenCL
OpenCL is a library for running programs in parallel, meaning that you can run many instances of the same program at the same time. I knew that if I wanted something extremely impressive that's what I would have to do. Graphics cards are designed to run in parallel, so that what I used to get the time you saw. I'm guessing that you'll want to run the code on your computer, so I'll point you to some resources to get your system ready. 

[This](https://github.com/KhronosGroup/OpenCL-Guide) is a github to get you introduced to OpenCL. It has a couple of pages for getting OpenCL installed.

If that's all working. You can 
