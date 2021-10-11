// copy from linux list
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ****************************************************************************

struct list_head {
  struct list_head *prev;
  struct list_head *next;
};

// ****************************************************************************

#define offset(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define container_of(ptr, type, member) ({			\
        const typeof( ((type *)0)->member ) *__mptr = (ptr);	\
        (type *)( (char *)__mptr - offset(type,member) );})

#define list_entry(ptr, type, member) container_of(ptr, type, member)

/**
 * list_for_each_entry_safe - iterate over list of given type safe against
 * removal of list entry
 * @pos:	the type * to use as a loop cursor.
 * @n:		another type * to use as temporary storage
 * @head:	the head for your list.
 * @member:	the name of the list_struct within the struct.
 */
#define list_for_each_entry_safe(pos, n, head, member)			\
	for (pos = list_entry((head)->next, typeof(*pos), member),	\
		n = list_entry(pos->member.next, typeof(*pos), member);	\
	     &pos->member != (head); 					\
	     pos = n, n = list_entry(n->member.next, typeof(*n), member))


/**
 * list_for_each_entry	-	iterate over list of given type
 * @pos:	the type * to use as a loop cursor.
 * @head:	the head for your list.
 * @member:	the name of the list_struct within the struct.
 */
#define list_for_each_entry(pos, head, member)				\
	for (pos = list_entry((head)->next, typeof(*pos), member);	\
	     tmc_mem_prefetch(pos->member.next, sizeof(*pos)), &pos->member != (head); 	\
	     pos = list_entry(pos->member.next, typeof(*pos), member))


// ****************************************************************************

static inline void
list_head_init(struct list_head *list)
{
  list->prev = list;
  list->next = list;
}

static inline int
list_empty(const struct list_head *head)
{
  return head->next == head;
}

static inline void
__list_add(struct list_head *new_node,
           struct list_head *prev,
           struct list_head *next)
{
  next->prev = new_node;
  new_node->next = next;
  new_node->prev = prev;
  prev->next = new_node;
}

static inline void
list_add_(struct list_head *new_node,
         struct list_head *head)
{
  __list_add(new_node, head, head->next);
}

static inline void
list_add_tail_(struct list_head *new_node,
              struct list_head *head)
{
  __list_add(new_node, head->prev, head);
}

static inline void
__list_del(struct list_head *prev,
           struct list_head *next)
{
  next->prev = prev;
  prev->next = next;
}

static inline void
list_del_(struct list_head *entry)
{
  __list_del(entry->prev, entry->next);
  entry->next = entry;
  entry->prev = entry;
}

// ****************************************************************************

#ifdef __cplusplus
}
#endif  // cplusplus

// ****************************************************************************
