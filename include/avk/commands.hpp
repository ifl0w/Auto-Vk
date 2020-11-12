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

	auto copy_buffer_to_another(avk::resource_reference<buffer_t> aSrcBuffer, avk::resource_reference<buffer_t> aDstBuffer, std::optional<vk::DeviceSize> aSrcOffset, std::optional<vk::DeviceSize> aDstOffset, std::optional<vk::DeviceSize> aDataSize)
	{
		auto impl = [aSrcBuffer, aDstBuffer, aSrcOffset, aDstOffset, aDataSize](resource_reference<command_buffer_t> aCommandBuffer){

			vk::DeviceSize dataSize{0};
			if (aDataSize.has_value()) {
				dataSize = aDataSize.value();
			}
			else {
				dataSize = aSrcBuffer->meta_at_index<buffer_meta>().total_size();
			}
			
	#ifdef _DEBUG
			{
				const auto& metaDataSrc = aSrcBuffer->meta_at_index<buffer_meta>();
				const auto& metaDataDst = aDstBuffer->meta_at_index<buffer_meta>();
				assert (aSrcOffset.value_or(0) + dataSize <= metaDataSrc.total_size());
				assert (aDstOffset.value_or(0) + dataSize <= metaDataDst.total_size());
				assert (aSrcOffset.value_or(0) + dataSize <= metaDataDst.total_size());
			}
	#endif
			
			auto copyRegion = vk::BufferCopy{}
				.setSrcOffset(aSrcOffset.value_or(0))
				.setDstOffset(aDstOffset.value_or(0))
				.setSize(dataSize);
			aCommandBuffer->handle().copyBuffer(aSrcBuffer->handle(), aDstBuffer->handle(), 1u, &copyRegion);
			
		};

		//// Sync before:
		//aSyncHandler.establish_barrier_before_the_operation(pipeline_stage::transfer, read_memory_access{memory_access::transfer_read_access});
		//// Sync after:
		//aSyncHandler.establish_barrier_after_the_operation(pipeline_stage::transfer, write_memory_access{memory_access::transfer_write_access});
		//// Finish him:
		//return aSyncHandler.submit_and_sync();

		return std::forward_as_tuple(
			// DESTINATION sync parameters to synchronize with whatever happened before:
			std::make_tuple(pipeline_stage::transfer, memory_access::transfer_read_access), // TODO: Could there be cases where also write-access must be synchronized?
			// The operation:
			impl,
			// SOURCE sync parameters to synchronize with whatever happens after:
			std::make_tuple(pipeline_stage::transfer, memory_access::transfer_write_access)
		);
	}
	
	auto copy_into_buffer(const void* aSrcDataPtr, resource_reference<buffer_t> aDstBuffer, size_t aMetaDataIndex, size_t aOffsetInBytes, size_t aDataSizeInBytes)
	{
		auto impl = [aSrcDataPtr, aDstBuffer, aMetaDataIndex, aOffsetInBytes, aDataSizeInBytes](resource_reference<command_buffer_t> aCommandBuffer){
			auto recursiveImpl = [](const void* bSrcDataPtr, resource_reference<buffer_t> bDstBuffer, size_t bMetaDataIndex, size_t bOffsetInBytes, size_t bDataSizeInBytes, resource_reference<command_buffer_t> bCommandBuffer, const auto& bCopyIntoBufferLambda){
				auto dataSize = static_cast<vk::DeviceSize>(bDataSizeInBytes);
				auto memProps = bDstBuffer->memory_properties();

#ifdef _DEBUG
				const auto& metaData = meta_at_index<buffer_meta>(aMetaDataIndex);
				assert(aOffsetInBytes + aDataSizeInBytes <= metaData.total_size()); // The fill operation would write beyond the buffer's size.
#endif

				// #1: Is our memory accessible from the CPU-SIDE? 
				if (avk::has_flag(memProps, vk::MemoryPropertyFlagBits::eHostVisible)) {
					auto scopedMapping = bDstBuffer->map_memory(mapping_access::write);
					memcpy(static_cast<uint8_t *>(scopedMapping.get()) + bOffsetInBytes, bSrcDataPtr, dataSize);
					return;
				}

				// #2: Otherwise, it must be on the GPU-SIDE!
				else {
					assert(avk::has_flag(memProps, vk::MemoryPropertyFlagBits::eDeviceLocal));

					// We have to create a (somewhat temporary) staging buffer and transfer it to the GPU
					// "somewhat temporary" means that it can not be deleted in this function, but only
					//						after the transfer operation has completed => handle via sync
					auto stagingBuffer = root::create_buffer(
						bDstBuffer->physical_device(), bDstBuffer->device(), bDstBuffer->allocator(),
						AVK_STAGING_BUFFER_MEMORY_USAGE,
						vk::BufferUsageFlagBits::eTransferSrc,
						generic_buffer_meta::create_from_size(dataSize)
					);
					bCopyIntoBufferLambda(bSrcDataPtr, avk::referenced(stagingBuffer), 0, 0, dataSize, bCommandBuffer, bCopyIntoBufferLambda); // Recurse into the other if-branch

					// Sync before:
					//TODO: aSyncHandler.establish_barrier_before_the_operation(pipeline_stage::transfer, read_memory_access{memory_access::transfer_read_access});

					// Operation:
					copy_buffer_to_another(avk::referenced(stagingBuffer), bDstBuffer, 0, static_cast<vk::DeviceSize>(bOffsetInBytes), dataSize, sync::with_barriers_into_existing_command_buffer(bCommandBuffer.get(), {}, {}));

					// Sync after:
					//TODO: aSyncHandler.establish_barrier_after_the_operation(pipeline_stage::transfer, write_memory_access{memory_access::transfer_write_access});

					// Take care of the lifetime handling of the stagingBuffer, it might still be in use:
					bCommandBuffer->set_custom_deleter([
						lOwnedStagingBuffer{ std::move(stagingBuffer) }
					]() { /* Nothing to do here, the buffers' destructors will do the cleanup, the lambda is just storing it. */ });
					
					// Finish him:
					//return aSyncHandler.submit_and_sync();			
				}
			};
			recursiveImpl(aSrcDataPtr, aDstBuffer, aMetaDataIndex, aOffsetInBytes, aDataSizeInBytes, std::move(aCommandBuffer), recursiveImpl);
		};
		return impl;
	}


	struct copy_source_for_pointer
	{
		auto copy_destination(resource_reference<buffer_t> aDstBuffer, std::optional<size_t> aMetaDataIndex, std::optional<size_t> aOffsetInBytes, std::optional<size_t> aDataSizeInBytes) const
		{
			return copy_into_buffer(mSrc, std::move(aDstBuffer), aMetaDataIndex.value_or(0), aOffsetInBytes.value_or(0), aDataSizeInBytes.value_or(aDstBuffer->meta_at_index<generic_buffer_meta>(0).total_size()));
		}
		
		const void* mSrc;
	};
	copy_source_for_pointer copy_source(const void* aSrcDataPtr) { return copy_source_for_pointer{ aSrcDataPtr }; }
		
	struct copy_source_for_reference
	{
		auto copy_destination(resource_reference<buffer_t> aDstBuffer, std::optional<size_t> aMetaDataIndex, std::optional<size_t> aOffsetInBytes, std::optional<size_t> aDataSizeInBytes) const
		{
			// TODO: use vkCmdCopyBuffer
			//return copy_into_buffer(mSrc, std::move(aDstBuffer), aMetaDataIndex.value_or(0), aOffsetInBytes.value_or(0), aDataSizeInBytes.value_or(aDstBuffer->meta_at_index<generic_buffer_meta>(0).total_size()));
		}
		
		resource_reference<buffer_t> mSrc;
	};
	copy_source_for_reference copy_source(resource_reference<buffer_t> aSrcBuffer) { return copy_source_for_reference{ std::move(aSrcBuffer) }; }
		
	struct copy_source_for_ownership
	{
		auto copy_destination(resource_reference<buffer_t> aDstBuffer, std::optional<size_t> aMetaDataIndex, std::optional<size_t> aOffsetInBytes, std::optional<size_t> aDataSizeInBytes) const
		{
			// TODO: use vkCmdCopyBuffer, handle the lifetime of mSrc in some way!
			//return unique_function<void(resource_reference<command_buffer_t>)>(....)
			//return copy_into_buffer(mSrc, std::move(aDstBuffer), aMetaDataIndex.value_or(0), aOffsetInBytes.value_or(0), aDataSizeInBytes.value_or(aDstBuffer->meta_at_index<generic_buffer_meta>(0).total_size()));
		}
		
		resource_ownership<buffer_t> mSrc;
	};
	copy_source_for_ownership copy_source(resource_ownership<buffer_t> aSrcBuffer) { return copy_source_for_ownership{ std::move(aSrcBuffer) }; }
		

}
