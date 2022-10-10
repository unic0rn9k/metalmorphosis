use crate::{BoxFuture, TaskHandle, TaskNode, Work, work, MorphicIO, buffer};

// Root node should return this, and should be used for allocation proceding.
// Parents should pass return value to childre, and then also keep one for reading afterwards.
trait Buffer{
    fn push<T: MorphicIO<'a>>(val: T) -> buffer::Alias<'a>;
}

pub trait Descriptor<'a> {
    fn children(&self) -> usize;
    fn node_depth_running_count(&self, ofset: usize) -> Vec<usize>{
        // This can probs be done more efficiently
        let mut sum = vec![ofset+1];
        for n in 0..self.children(){
            sum.append(&mut self.child(n).unwrap().node_depth_running_count(ofset+1+n))
        }
        sum
    }

    fn output_size(&self) -> usize;
    fn buffer(&self) -> Vec<u8>{
        vec![]
    }

    fn child(&self, idx: usize) -> Option<Box<dyn Descriptor<'a>>>;
    fn task(&self, handle: &TaskHandle<'a, ()>) -> Work<'a>;

    fn load(&self, buffer: &mut [TaskNode<'a>], handle: &TaskHandle<'a, ()>) {
        debug_assert!(
            buffer.len() == self.sub_nodes() + 1,
            "{} != {}",
            buffer.len(),
            self.sub_nodes() + 1
        );

        // This should be done in for loop.
        buffer[0] = TaskNode::<'a> {
            future: self.task().extremely_unsafe_type_conversion(),
            children: self.children(),
            parent: hande.this_node,
            output: buffer::Source::<'a, ()>::uninit().alias()
        };

        let mut child_addr = 1;

        for n in 0..self.sub_nodes() {
            let child = if let Some(n) = self.child(n) {
                n
            } else {
                return;
            };
            let child_sub_nodes = self.child(n).unwrap().sub_nodes();
            child.load(
                &mut buffer[child_addr..child_addr + child_sub_nodes + 1],
                ofset + child_addr,
            );
            child_addr += child_sub_nodes + 1;
        }
    }

    fn graph(&self, ofset: usize) -> Vec<TaskNode<'a>> {
        let mut buffer = Vec::with_capacity(self.sub_nodes() + 1);
        self.load(&mut buffer[..], ofset);
        buffer
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
            work(async move{
                todo!()
             })
        }
    }

    let bruh = Nop(vec![
        Nop(vec![Nop(vec![]), Nop(vec![Nop(vec![]), Nop(vec![])])]),
        Nop(vec![Nop(vec![]), Nop(vec![])]),
    ]);
    assert_eq!(bruh.sub_nodes(), 8);
    bruh.graph(0);
}
