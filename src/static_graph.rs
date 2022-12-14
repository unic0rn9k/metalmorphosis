// https://news.ycombinator.com/item?id=20512490
// interesting...

use crate::{buffer, MorphicIO, TaskHandle, TaskNode, Work};

// # Cache Pre-allocation
//
// if !this.is_prealloc(){
//  let mut prealloc = buffer::new(this.child_prealloc);
//  self.prealloc = &mut prealloc as *mut Buffer
// }else{
//  self.prealloc = parent.prealloc;
// }
//
// let alloc = prealloc::Static::<23>();
// handle.with_preallocator(&mut alloc)
//
// When branch is called, self.prealloc should be used (if this.is_prealloc) for allocating buffer::Alias'
//
//
// # Static graph
//
// n = task_graph.len() // not this_node!
// nodes -> task_graph::reserve
// bytes -> buffer::new
//
// for node{
//  output = buffer.alloc(node::T)
//  node(n, output) -> task_graph.insert(node.this_node + n)
//  node(n, buffer) // make node insert its children relative to this_node and n
// }
//

// Root node should return this, and should be used for allocation proceding.
// Parents should pass return value to childre, and then also keep one for reading afterwards.
trait Buffer<'a> {
    // En ring buffer, der tjekker om data den overwriter, er blevet dealokeret.
    // Den kan finde ud af om det er deallokeret ved at kigge om de nodes der ejer outputtet til
    // den data der der fylder den mindste mængde bytes der passer til den nye data.
    // fordi det er en ring buffer vil de ældste nodes blive tjekket først.
    fn alloc<T: MorphicIO<'a>>(&mut self) -> buffer::Alias<'a>;
    fn new(capacity_in_bytes: usize) -> Self;
}

pub trait Descriptor<'a> {
    fn children(&self) -> usize;

    /// It's likely a good idea to overwrite this, if you know exactly how many nodes are in the graph.
    ///
    /// ```text
    ///   O
    ///  /\
    /// O  O
    ///  \
    ///   O
    /// ```
    ///
    /// For example, the graph above should return 3.
    fn sub_nodes(&self) -> usize {
        let mut sum = self.children();
        for n in 0..self.children() {
            sum += self.child(n).sub_nodes();
        }
        sum
    }

    /// Should return the sum of the output sizes of all nodes in the graph.
    fn output_size(&self) -> usize {
        todo!()
    }

    fn child(&self, idx: usize) -> Option<Box<dyn Descriptor<'a>>>;

    /// This is where you put the actual funtion of the node.
    fn task(&self, handle: &TaskHandle<'a, ()>) -> Work<'a>;

    fn load(&self, buffer: &mut Vec<TaskNode<'a>>, handle: &TaskHandle<'a, ()>) {
        // This should be done in for loop.
        buffer.push(TaskNode::<'a> {
            future: self.task().extremely_unsafe_type_conversion(),
            children: self.children(),
            parent: handle.this_node, // +ofset ?
            output: buffer::Source::<'a, ()>::uninit().alias(),
        });

        let mut child_addr = 1;

        for n in 0..self.sub_nodes() {
            let child = if let Some(n) = self.child(n) {
                n
            } else {
                return;
            };
            let child_sub_nodes = self.child(n).unwrap().sub_nodes();
            child.load(&mut buffer, ofset + child_addr);
            child_addr += child_sub_nodes + 1;
        }
    }

    fn graph(&self, ofset: usize) -> Vec<TaskNode<'a>> {
        let mut output_buffer = Self::Buffer::new(self.output_size());
        let mut node_buffer = Vec::with_capacity(self.sub_nodes() + 1);
        self.load(&mut node_buffer, /*handle, */ ofset);
        node_buffer
    }
}

#[test]
fn nop_graph() {
    #[derive(Clone)]
    pub struct Nop(Vec<Nop>);

    impl<'a> Descriptor<'a> for Nop {
        fn child(&self, idx: usize) -> Option<Box<dyn Descriptor<'a>>> {
            match self.0.get(idx) {
                Some(n) => Some(Box::new(n.clone())),
                _ => None,
            }
        }

        fn children(&self) -> usize {
            self.0.len()
        }

        fn task(&self) -> Work<'a> {
            crate::work(async move { todo!() })
        }
    }

    let bruh = Nop(vec![
        Nop(vec![Nop(vec![]), Nop(vec![Nop(vec![]), Nop(vec![])])]),
        Nop(vec![Nop(vec![]), Nop(vec![])]),
    ]);
    assert_eq!(bruh.sub_nodes(), 8);
    bruh.graph(0);
}
