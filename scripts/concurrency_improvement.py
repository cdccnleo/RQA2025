#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发处理能力提升脚本
"""

import asyncio
import threading
import concurrent.futures
import multiprocessing
import time
from queue import Queue
import json


class AsyncTaskProcessor:
    """异步任务处理器"""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process_task(self, task_id, data):
        """异步处理任务"""
        async with self.semaphore:
            start_time = time.time()

            # 模拟异步I/O操作
            await asyncio.sleep(0.1)

            # 模拟计算密集型操作
            result = self.compute_intensive_task(data)

            end_time = time.time()

            return {
                "task_id": task_id,
                "result": result,
                "processing_time": end_time - start_time
            }

    def compute_intensive_task(self, data):
        """计算密集型任务"""
        result = 0
        for i in range(len(data)):
            result += data[i] * data[i]
        return result


class ThreadPoolProcessor:
    """线程池处理器"""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def process_tasks(self, tasks):
        """处理多个任务"""
        start_time = time.time()

        futures = [self.executor.submit(self.process_single_task, task) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()

        return {
            "processor_type": "thread_pool",
            "max_workers": self.max_workers,
            "total_tasks": len(tasks),
            "total_time": end_time - start_time,
            "tasks_per_second": len(tasks) / (end_time - start_time),
            "results": results
        }

    def process_single_task(self, task):
        """处理单个任务"""
        start_time = time.time()

        # 模拟处理逻辑
        time.sleep(0.1)
        result = task["id"] * 1000

        end_time = time.time()

        return {
            "task_id": task["id"],
            "result": result,
            "processing_time": end_time - start_time
        }


def test_async_processing(num_tasks=100):
    """测试异步处理"""
    print(f"测试异步处理能力 ({num_tasks}个任务)...")

    processor = AsyncTaskProcessor(max_workers=10)

    async def main():
        tasks = []
        for i in range(num_tasks):
            task = processor.process_task(i, [i] * 100)
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "processor_type": "async",
            "total_tasks": num_tasks,
            "total_time": end_time - start_time,
            "tasks_per_second": num_tasks / (end_time - start_time),
            "avg_processing_time": sum(r["processing_time"] for r in results) / len(results),
            "results_count": len(results)
        }

    return asyncio.run(main())


def test_thread_pool_processing(num_tasks=100):
    """测试线程池处理"""
    print(f"测试线程池处理能力 ({num_tasks}个任务)...")

    processor = ThreadPoolProcessor(max_workers=8)

    tasks = [{"id": i} for i in range(num_tasks)]

    return processor.process_tasks(tasks)


def test_multiprocessing(num_tasks=50):
    """测试多进程处理"""
    print(f"测试多进程处理能力 ({num_tasks}个任务)...")

    start_time = time.time()

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_multiprocessing_task, range(num_tasks))

    end_time = time.time()

    return {
        "processor_type": "multiprocessing",
        "total_tasks": num_tasks,
        "total_time": end_time - start_time,
        "tasks_per_second": num_tasks / (end_time - start_time),
        "results_count": len(results)
    }


def process_multiprocessing_task(task_id):
    """多进程任务处理"""
    # 模拟CPU密集型计算
    result = 0
    for i in range(100000):
        result += i
    return {"task_id": task_id, "result": result}


def test_work_queue_processing(num_tasks=100):
    """测试工作队列处理"""
    print(f"测试工作队列处理能力 ({num_tasks}个任务)...")

    task_queue = Queue()
    result_queue = Queue()

    # 启动工作线程
    workers = []
    for i in range(4):
        worker = threading.Thread(
            target=worker_function,
            args=(task_queue, result_queue, i)
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    start_time = time.time()

    # 添加任务到队列
    for i in range(num_tasks):
        task_queue.put({"id": i, "data": [i] * 10})

    # 添加结束标记
    for _ in range(4):
        task_queue.put(None)

    # 收集结果
    results = []
    for _ in range(num_tasks):
        result = result_queue.get()
        results.append(result)

    end_time = time.time()

    return {
        "processor_type": "work_queue",
        "total_tasks": num_tasks,
        "total_time": end_time - start_time,
        "tasks_per_second": num_tasks / (end_time - start_time),
        "results_count": len(results)
    }


def worker_function(task_queue, result_queue, worker_id):
    """工作线程函数"""
    while True:
        task = task_queue.get()
        if task is None:
            break

        # 处理任务
        start_time = time.time()
        time.sleep(0.05)  # 模拟处理时间
        result = task["id"] * 1000
        end_time = time.time()

        result_data = {
            "task_id": task["id"],
            "worker_id": worker_id,
            "result": result,
            "processing_time": end_time - start_time
        }

        result_queue.put(result_data)


def main():
    """主函数"""
    print("开始并发处理能力提升测试...")

    concurrency_results = {
        "test_time": time.time(),
        "tests": []
    }

    try:
        # 测试异步处理
        print("\n1. 测试异步处理:")
        async_result = test_async_processing(num_tasks=50)
        concurrency_results["tests"].append(async_result)
        print(f"   任务数: {async_result['total_tasks']}")
        print(f"   每秒处理: {async_result['tasks_per_second']:.2f}个任务")

        # 测试线程池处理
        print("\n2. 测试线程池处理:")
        thread_result = test_thread_pool_processing(num_tasks=50)
        concurrency_results["tests"].append(thread_result)
        print(f"   任务数: {thread_result['total_tasks']}")
        print(f"   每秒处理: {thread_result['tasks_per_second']:.2f}个任务")

        # 测试多进程处理
        print("\n3. 测试多进程处理:")
        process_result = test_multiprocessing(num_tasks=25)
        concurrency_results["tests"].append(process_result)
        print(f"   任务数: {process_result['total_tasks']}")
        print(f"   每秒处理: {process_result['tasks_per_second']:.2f}个任务")

        # 测试工作队列处理
        print("\n4. 测试工作队列处理:")
        queue_result = test_work_queue_processing(num_tasks=50)
        concurrency_results["tests"].append(queue_result)
        print(f"   任务数: {queue_result['total_tasks']}")
        print(f"   每秒处理: {queue_result['tasks_per_second']:.2f}个任务")

    except Exception as e:
        print(f"测试过程中出错: {e}")

    # 保存结果
    with open('concurrency_improvement_results.json', 'w', encoding='utf-8') as f:
        json.dump(concurrency_results, f, indent=2, ensure_ascii=False)

    print("\n并发处理能力提升测试完成，结果已保存到 concurrency_improvement_results.json")

    return concurrency_results


if __name__ == '__main__':
    main()
