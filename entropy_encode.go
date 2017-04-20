package guetzli_patapon

import "sort"

// A node of a Huffman tree.
type HuffmanTree struct {
	total_count_          uint32
	index_left_           int16
	index_right_or_value_ int16
}

func NewHuffmanTree(count uint32, left, right int16) *HuffmanTree {
	return &HuffmanTree{
		total_count_:          count,
		index_left_:           left,
		index_right_or_value_: right,
	}
	// TODO PATAPON: maybe return value, instead of pointer?
}

func SetDepth(p0 int, pool []HuffmanTree, depth []uint8, max_depth int) bool {
	var stack [17]int
	level := 0
	p := p0
	assert(max_depth <= 16)
	stack[0] = -1
	for {
		if pool[p].index_left_ >= 0 {
			level++
			if level > max_depth {
				return false
			}
			stack[level] = int(pool[p].index_right_or_value_)
			p = int(pool[p].index_left_)
			continue
		} else {
			depth[pool[p].index_right_or_value_] = uint8(level)
		}
		for level >= 0 && stack[level] == -1 {
			level--
		}
		if level < 0 {
			return true
		}
		p = stack[level]
		stack[level] = -1
	}
}

// Sort the root nodes, least popular first.
func SortHuffmanTree(v0, v1 *HuffmanTree) bool {
	if v0.total_count_ != v1.total_count_ {
		return v0.total_count_ < v1.total_count_
	}
	return v0.index_right_or_value_ > v1.index_right_or_value_
}

// This function will create a Huffman tree.
//
// The catch here is that the tree cannot be arbitrarily deep.
// Brotli specifies a maximum depth of 15 bits for "code trees"
// and 7 bits for "code length code trees."
//
// count_limit is the value that is to be faked as the minimum value
// and this minimum value is raised until the tree matches the
// maximum length requirement.
//
// This algorithm is not of excellent performance for very long data blocks,
// especially when population counts are longer than 2**tree_limit, but
// we are not planning to use this with extremely long blocks.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
func CreateHuffmanTree(data []uint32,
	length int,
	tree_limit int,
	tree []HuffmanTree,
	depth []uint8) {
	// For block sizes below 64 kB, we never need to do a second iteration
	// of this loop. Probably all of our block sizes will be smaller than
	// that, so this loop is mostly of academic interest. If we actually
	// would need this, we would be better off with the Katajainen algorithm.
	for count_limit := uint32(1); ; count_limit *= 2 {
		n := 0
		for i := length; i != 0; {
			i--
			if data[i] != 0 {
				count := std_maxUint32(data[i], count_limit)
				tree[n] = *NewHuffmanTree(count, -1, int16(i))
				n++
			}
		}

		if n == 1 {
			depth[tree[0].index_right_or_value_] = 1 // Only one element.
			break
		}

		sort.Slice(tree[:n], func(i, j int) bool {
			return SortHuffmanTree(&tree[i], &tree[j])
		})

		// The nodes are:
		// [0, n): the sorted leaf nodes that we start with.
		// [n]: we add a sentinel here.
		// [n + 1, 2n): new parent nodes are added here, starting from
		//              (n+1). These are naturally in ascending order.
		// [2n]: we add a sentinel at the end as well.
		// There will be (2n+1) elements at the end.
		sentinel := *NewHuffmanTree(^uint32(0), -1, -1)
		tree[n] = sentinel
		tree[n+1] = sentinel

		i := 0     // Points to the next leaf node.
		j := n + 1 // Points to the next non-leaf node.
		for k := n - 1; k != 0; k-- {
			var left, right int
			if tree[i].total_count_ <= tree[j].total_count_ {
				left = i
				i++
			} else {
				left = j
				j++
			}
			if tree[i].total_count_ <= tree[j].total_count_ {
				right = i
				i++
			} else {
				right = j
				j++
			}

			// The sentinel node becomes the parent node.
			j_end := 2*n - k
			tree[j_end].total_count_ =
				tree[left].total_count_ + tree[right].total_count_
			tree[j_end].index_left_ = int16(left)
			tree[j_end].index_right_or_value_ = int16(right)

			// Add back the last sentinel node.
			tree[j_end+1] = sentinel
		}
		if SetDepth(int(2*n-1), tree, depth, tree_limit) {
			/* We need to pack the Huffman tree in tree_limit bits. If this was not
			   successful, add fake entities to the lowest values and retry. */
			break
		}
	}
}
