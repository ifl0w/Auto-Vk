#pragma once
#include <avk/avk.hpp>

namespace avk
{
	class command_buffer_t;
	using command_buffer = avk::owning_resource<command_buffer_t>;
	
	class commands
	{
		using rec_cmd = std::function<void(avk::command_buffer_t&)>;
		using sync_data = std::tuple<avk::pipeline_stage, avk::memory_access>;
		
	public:
		/** You're not supposed to store an instance of commands anywhere. */
		commands() = delete;
		commands(commands&&) noexcept = default;
		commands(const commands&) = delete;
		commands& operator=(commands&&) noexcept = default;
		commands& operator=(const commands&) = delete;
		~commands() = default;

		template <typename F>
		commands(avk::pipeline_stage aPreSyncDst, avk::memory_access aPreAccessDst, F&& aCommands, avk::pipeline_stage aPostSyncSrc, avk::memory_access aPostAccessSrc)
			: mRecordCommands{ std::forward(aCommands) }
			, mPreSync{ aPreSyncDst, aPreAccessDst }
			, mPostSync{ aPostSyncSrc, aPostAccessSrc }
		{ }
		
		void record_into(command_buffer_t& aCommandBuffer)
		{
			mRecordCommands(aCommandBuffer);
		}
		
		rec_cmd mRecordCommands;
		sync_data mPreSync;
		sync_data mPostSync;
	};


//	auto copy_into_buffer(resource_reference<buffer_t> aTargetBuffer, const void* aDataPtr, size_t aMetaDataIndex, size_t aOffsetInBytes, size_t aDataSizeInBytes)
//	{
//		return[
//			aTargetBuffer,
//			aDataPtr,
//			aMetaDataIndex,
//			aOffsetInBytes,
//			aDataSizeInBytes
//		](resource_reference<command_buffer_t> aCommandBuffer){
//			auto dataSize = static_cast<vk::DeviceSize>(aDataSizeInBytes);
//			auto memProps = aTargetBuffer->memory_properties();
//
//#ifdef _DEBUG
//			const auto& metaData = meta_at_index<buffer_meta>(aMetaDataIndex);
//			assert(aOffsetInBytes + aDataSizeInBytes <= metaData.total_size()); // The fill operation would write beyond the buffer's size.
//#endif
//
//			// #1: Is our memory accessible from the CPU-SIDE? 
//			if (avk::has_flag(memProps, vk::MemoryPropertyFlagBits::eHostVisible)) {
//				auto scopedMapping = aTargetBuffer->map_memory(mapping_access::write);
//				memcpy(static_cast<uint8_t *>(scopedMapping.get()) + aOffsetInBytes, aDataPtr, dataSize);
//				return;
//			}
//
//			// #2: Otherwise, it must be on the GPU-SIDE!
//			else {
//				assert(avk::has_flag(memProps, vk::MemoryPropertyFlagBits::eDeviceLocal));
//
//				// We have to create a (somewhat temporary) staging buffer and transfer it to the GPU
//				// "somewhat temporary" means that it can not be deleted in this function, but only
//				//						after the transfer operation has completed => handle via sync
//				auto stagingBuffer = root::create_buffer(
//					aTargetBuffer->physical_device(), aTargetBuffer->device(), aTargetBuffer->allocator(),
//					AVK_STAGING_BUFFER_MEMORY_USAGE,
//					vk::BufferUsageFlagBits::eTransferSrc,
//					generic_buffer_meta::create_from_size(dataSize)
//				);
//				copy_into_buffer(avk::referenced(stagingBuffer), aDataPtr, 0, 0, dataSize)(aCommandBuffer); // Recurse into the other if-branch
//
//				// Sync before:
//				//TODO: aSyncHandler.establish_barrier_before_the_operation(pipeline_stage::transfer, read_memory_access{memory_access::transfer_read_access});
//
//				// Operation:
//				copy_buffer_to_another(avk::referenced(stagingBuffer), aTargetBuffer, 0, static_cast<vk::DeviceSize>(aOffsetInBytes), dataSize, sync::with_barriers_into_existing_command_buffer(aCommandBuffer.get(), {}, {}));
//
//				// Sync after:
//				//TODO: aSyncHandler.establish_barrier_after_the_operation(pipeline_stage::transfer, write_memory_access{memory_access::transfer_write_access});
//
//				// Take care of the lifetime handling of the stagingBuffer, it might still be in use:
//				aCommandBuffer->set_custom_deleter([
//					lOwnedStagingBuffer{ std::move(stagingBuffer) }
//				]() { /* Nothing to do here, the buffers' destructors will do the cleanup, the lambda is just storing it. */ });
//				
//				// Finish him:
//				//return aSyncHandler.submit_and_sync();			
//			}
//		};
//		
//	}
	
}
