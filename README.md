<div align="center">
<h1> metalmorphosis </h1>
</div>

Distributed async runtime in rust, with a focus on being able to build computation graphs (specifically auto-diff).

examples can be found in examples directory.

### Weird place to have a todo list...
- Maybe rename MorphicIO back to Distributed or distributable.
- examples/math.rs (AutoDiff)
- src/network.rs (distribute that bitch)
- I removed wakers again
- Mixed static and dynamic graphs. (Describe location of static node based on displacement from dynamic parent node)
- Node caching

### Project timeline
0. Auto-diff graph (linear algebra mby)
1. multi-threaded (Static graphs, node caching)
2. distributed (mio and buffer/executor changes)
3. Route optimization (also when should caching occur? maybe just tell explicitly when :/)

### Distributed pointers
Function side-effects are very inefficient on a distributed system,
as there is no way to directly mutate data on another device.

The easiest way to handle data return might be with distributed side-effects tho.
Just make buffer::Alias serializable and contain a machine-id.
Then when you want to write to it, it might just send the pointer and data to the machine with the id,
which will then write the data.
This will of course likely only work if the data is in the serialized format.

it should be possible to do *Prefetching* of distributed pointer values.
Meaning if we know that 'this device' is gonna read from 'other device',
and other device already has the value ready.
then it would make sense to schedule a read from other device,
even tho this device doesn't need the value yet.

## Implement me
https://play.rust-lang.org/?version=nightly&mode=debug&edition=2021&code=use%20std%3A%3Aops%3A%3A*%3B%0A%0Astruct%20TaskHandle%3B%0A%0Atrait%20TaskHandleProvider%3C%27a%3E%7B%0A%20%20%20%20fn%20handle(%26%27a%20mut%20self)%20-%3E%20%26%27a%20mut%20TaskHandle%3B%0A%7D%0A%0Aimpl%3C%27a%3E%20TaskHandleProvider%3C%27a%3E%20for%20TaskHandle%7B%0A%20%20%20%20%23%5Binline(always)%5D%0A%20%20%20%20fn%20handle(%26%27a%20mut%20self)%20-%3E%20%26%27a%20mut%20Self%7B%0A%20%20%20%20%20%20%20%20self%0A%20%20%20%20%7D%0A%7D%0A%0Aimpl%3C%27a%2C%20I%3A%20Iterator%3CItem%3D%26%27a%20mut%20TaskHandle%3E%3E%20TaskHandleProvider%3C%27a%3E%20for%20I%7B%0A%20%20%20%20%23%5Binline(always)%5D%0A%20%20%20%20fn%20handle(%26%27a%20mut%20self)%20-%3E%20%26%27a%20mut%20TaskHandle%7B%0A%20%20%20%20%20%20%20%20self.next().unwrap()%0A%20%20%20%20%7D%0A%7D%0A%0Astruct%20IntoTask%3B%0A%0Aimpl%20IntoTask%7B%0A%20%20%20%20%2F%2F%20Maybe%20this%20should%20return%20an%20Iterator%3Cimpl%20Fn%3E%0A%20%20%20%20%2F%2F%20https%3A%2F%2Fdocs.rs%2Ffutures%2Flatest%2Ffutures%2Fstream%2Ftrait.Stream.html%0A%20%20%20%20fn%20task%3C%27a%2C%20H%3A%20TaskHandleProvider%3C%27a%3E%3E(%26self%2C%20handle%3A%20H)-%3E%20impl%20Fn()%7B%0A%20%20%20%20%20%20%20%20move%20%7C%7Cprintln!(%22ok%22)%0A%20%20%20%20%7D%0A%7D%0A%0Afn%20spawn%3CF%3A%20Fn()%2C%20T%3A%20Fn(TaskHandle)-%3EF%3E(task%3A%20T)%7B%0A%20%20%20%20task(TaskHandle)()%0A%7D%0A%0Afn%20main()%7B%0A%20%20%20%20let%20a%20%3D%20IntoTask%3B%0A%20%20%20%20for%20n%20in%200..10%7B%0A%20%20%20%20%20%20%20%20spawn(%7Chandle%7Ca.task(handle))%0A%20%20%20%20%7D%0A%7D%0A%0A%0A%2F%2F%20Synchronous%20reusability%20is%20the%20same%20as%3A%0A%2F%2F%20fn(task)%7B%0A%2F%2F%20%20%20parent.send(handle.branch(task).hint(KeepLocal))%3B%0A%2F%2F%20%20%20parent.poll%0A%2F%2F%20%7D%0A%2F%2F%0A%2F%2F%20It%20would%20ever%20make%20sence%20to%20do%20this%20over%20a%20network%2C%0A%2F%2F%20as%20the%20executor%20would%20never%20be%20able%20to%20re-distribute%20the%20task%20to%20a%20new%20device%0A%2F%2F%20whihch%20itself%20would%20never%20have%20enough%20overhead%20(compared%20to%20the%20networking%20of%20sending%20data)%0A%2F%2F%20to%20actually%20justify%20not%20doing%20it.%0A%2F%2F%0A%2F%2F%20Thus%20it%20would%20always%20be%20better%20to%20just%20branch%20in%20a%20loop.%0A%2F%2F%20%0A%2F%2F%20Given%20that%20it%20can%20be%20avoided%20to%20run%20setup%20multiple%20times.
